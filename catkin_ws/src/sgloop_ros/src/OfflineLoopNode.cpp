#include <sstream>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"

#include "open3d_conversions/open3d_conversions.h"

#include "mapping/SemanticMapping.h"
#include "tools/Tools.h"
#include "tools/IO.h"
#include "tools/Eval.h"
#include "tools/TicToc.h"
#include "tools/g3reg_api.h"
#include "sgloop/Graph.h"
#include "sgloop/SGNet.h"
#include "sgloop/ShapeEncoder.h"
#include "sgloop/LoopDetector.h"

#include "registration/Prune.h"
#include "communication/Communication.h"

#include "Visualization.h"

struct GraphMsg{
    sensor_msgs::PointCloud2 global_cloud;
}ref_msgs, src_msgs;

float check_two_points_vec(const std::vector<Eigen::Vector3d> src_points,
                        const std::vector<Eigen::Vector3d> ref_points)
{
    float dist = 0.0;
    for (int i=0;i<src_points.size();i++){
        dist += (src_points[i]-ref_points[i]).norm();
    }
    return dist;
}

/// @brief  Check the labels of two points vec.
/// @return The number of matched labels.
int check_points_labels(const std::vector<uint32_t> &src_labels, const std::vector<uint32_t> &ref_labels)
{
    assert(src_labels.size()==ref_labels.size());
    int count = 0;
    for (int i=0;i<src_labels.size();i++){
        if(src_labels[i]==ref_labels[i])
            count++;
    }
    return count;
}

int main(int argc, char **argv)
{
    using namespace fmfusion;
    using namespace open3d::utility;

    ros::init(argc, argv, "LoopNode");
    ros::NodeHandle n;
    ros::NodeHandle nh_private("~");

    // Settings
    std::string config_file, weights_folder;
    std::string ref_scene_dir, src_scene_dir;
    std::string LOCAL_AGENT, REMOTE_AGENT;

    bool set_cfg = nh_private.getParam("cfg_file", config_file);
    bool set_wegith_folder = nh_private.getParam("weights_folder", weights_folder);
    bool set_ref_scene = nh_private.getParam("ref_scene_dir", ref_scene_dir);
    bool set_src_scene = nh_private.getParam("src_scene_dir", src_scene_dir);
    bool set_local_agent = nh_private.getParam("local_agent", LOCAL_AGENT);
    bool set_remote_agent = nh_private.getParam("remote_agent", REMOTE_AGENT);

    std::string output_folder = nh_private.param("output_folder", std::string(""));
    std::string frame_id = nh_private.param("frame_id", std::string("world"));
    std::string gt_file = nh_private.param("gt_file", std::string(""));
    int visualization = nh_private.param("visualization", 0);
    bool prune_instances = nh_private.param("prune_instances", true);
    bool fused = nh_private.param("fused",true);
    bool test_communication = nh_private.param("test_communication", false);
    int broadcast_times = nh_private.param("broadcast_times", 2);
    bool dense_msg = nh_private.param("dense_msg", true);
    bool icp_refine = nh_private.param("icp_refine", true);
    bool early_stop = nh_private.param("early_stop", false); // Stop after telecom
    std::string ref_agent_name = "agentB";

    // Publisher
    Visualization::Visualizer viz(n,nh_private,{ref_agent_name});
    // ros::Subscriber ref_sub = nh_private.subscribe("coarse_graph",1000);
    std::string src_name = *filesystem::GetPathComponents(src_scene_dir).rbegin();
    std::string ref_name = *filesystem::GetPathComponents(ref_scene_dir).rbegin();
    std::string pair_name = src_name+"-"+ref_name;

    assert(set_cfg && set_wegith_folder);
    assert(set_ref_scene && set_src_scene);

    std::cout<<"config file: "<<config_file<<std::endl;

    // Load Config
    Config *sg_config = utility::create_scene_graph_config(config_file, true);
    if (sg_config==nullptr){
        ROS_WARN("Failed to create scene graph config.");
        return 0;
    }

    // Declare reference data
    std::vector<InstanceId> ref_names, src_names;
    std::vector<InstancePtr> ref_instances, src_instances;    
    fmfusion::o3d_utility::Timer timer;
    DataDict ref_data_dict;
    std::shared_ptr<SemanticMapping> ref_mapping = std::make_shared<SemanticMapping>(
        SemanticMapping(sg_config->mapping_cfg,sg_config->instance_cfg));
    std::shared_ptr<Graph> ref_graph = std::make_shared<Graph>(sg_config->graph);

    ref_mapping->load(ref_scene_dir);
    
    // Active graph
    fmfusion::TicToc tictoc;
    auto src_mapping = std::make_shared<SemanticMapping>(SemanticMapping(sg_config->mapping_cfg,sg_config->instance_cfg));
    src_mapping->load(src_scene_dir);
    src_mapping->extract_bounding_boxes();
    src_mapping->export_instances(src_names,src_instances);
    std::cout<<"prepare src scene: "<<src_name
        << ". it takes "<< tictoc.toc()<<"ms \n";

    auto src_graph = std::make_shared<Graph>(sg_config->graph);
    src_graph->initialize(src_instances);
    src_graph->construct_edges();
    src_graph->construct_triplets();
    fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();

    // Encode Src Graph
    auto loop_detector = std::make_shared<fmfusion::LoopDetector>(fmfusion::LoopDetector(
                        sg_config->loop_detector,
                        sg_config->shape_encoder, 
                        sg_config->sgnet, 
                        weights_folder,
                        0,
                        {ref_agent_name}));
    timer.Start();
    loop_detector->encode_src_scene_graph(src_graph->get_const_nodes());
    timer.Stop();
    std::cout<<"Encode ref graph takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";

    if (test_communication){
        ROS_WARN("Test communication");
        SgCom::Communication com_server(n, nh_private,LOCAL_AGENT,{REMOTE_AGENT});

        timer.Start();
        int Ns, Ds;
        std::vector<std::vector<float>> sent_coarse_feats_vec;
        fmfusion::DataDict sent_explicit_nodes;

        loop_detector->get_active_node_feats(sent_coarse_feats_vec, Ns, Ds);
        sent_explicit_nodes = src_graph->extract_data_dict(!dense_msg);
        timer.Stop();
        std::cout<<"Extract src data dict: "<<Ns<<"x"<<Ds<<". "
                    <<"It takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";

        for (int k=0;k<broadcast_times;k++){        
            timer.Start();
            // com_server.broadcast_coarse_graph(30,
            //                                     sent_explicit_nodes.instances,
            //                                     sent_explicit_nodes.centroids,
            //                                     Ns,Ds,
            //                                     sent_coarse_feats_vec);
            com_server.broadcast_dense_graph(30,
                                            sent_explicit_nodes.nodes,
                                            sent_explicit_nodes.instances,
                                            sent_explicit_nodes.centroids,
                                            sent_coarse_feats_vec,
                                            sent_explicit_nodes.xyz,
                                            sent_explicit_nodes.labels);
            timer.Stop();
            std::cout<<"Broadcast takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";

            ros::Duration(0.5).sleep();
            ros::spinOnce();

            // Vec->Tensor
            const SgCom::AgentDataDict received_agent_nodes = com_server.get_remote_agent_data(REMOTE_AGENT);
            if(received_agent_nodes.frame_id>=0)
            {
                int Nr = received_agent_nodes.N;
                int Dr = received_agent_nodes.D;
 
                // update
                torch::Tensor empty_tensor = torch::empty({0,0});

                loop_detector->subscribe_ref_coarse_features(ref_agent_name,
                                                            received_agent_nodes.received_timestamp, 
                                                            received_agent_nodes.features_vec,
                                                            empty_tensor);
                int update_nodes_count = ref_graph->subscribe_coarse_nodes(received_agent_nodes.received_timestamp,
                                                    received_agent_nodes.nodes,
                                                    received_agent_nodes.instances,
                                                    received_agent_nodes.centroids);

                int update_pts_count = 0;
                if(received_agent_nodes.X>0)
                    update_pts_count   = ref_graph->subscribde_dense_points(received_agent_nodes.received_timestamp,
                                                        received_agent_nodes.xyz,
                                                        received_agent_nodes.labels);
                ref_data_dict = ref_graph->extract_data_dict();

                std::cout<<"Update "<<update_nodes_count<<"/"
                        << received_agent_nodes.N<<" received nodes, "
                        << update_pts_count<<"/"
                        <<received_agent_nodes.xyz.size()<<" received points."<<std::endl;

                // Verify
                if(Ns==Nr){
                    std::cout<<"Checking pub and received tensor \n";
                    torch::Tensor raw_features = loop_detector->get_active_node_feats().to(torch::kCPU);
                    torch::Tensor received_tensor_bk = loop_detector->get_ref_node_feats(ref_agent_name).to(torch::kCPU);

                    std::cout<<"["<<LOCAL_AGENT<<"] received bk tensor close:"
                            <<torch::allclose(raw_features, received_tensor_bk , 1e-5)<<std::endl;
                    
                    //
                    int Xs = src_data_dict.xyz.size();
                    int Xr = received_agent_nodes.xyz.size();
                    std::cout<<"Comparing "<< Xs<<" src points and "
                            <<Xr<<" received points.\n";
                    
                    float dist = check_two_points_vec(src_data_dict.xyz, received_agent_nodes.xyz);
                    int consist_count = check_points_labels(src_data_dict.labels, received_agent_nodes.labels);

                    ROS_WARN("Check two points vec diff: %.3f, %d/%d consistently labeled", dist, consist_count, Xs);
                }
            }
        }

        // return 0;
    }
    else{// Load and Encode Ref Graph
        ref_mapping->extract_bounding_boxes();
        ref_mapping->export_instances(ref_names,ref_instances);        
        ref_graph->initialize(ref_instances);
        ref_graph->construct_edges();
        ref_graph->construct_triplets();

        timer.Start();                        
        loop_detector->encode_ref_scene_graph(ref_agent_name,ref_graph->get_const_nodes());
        timer.Stop();
        ref_data_dict = ref_graph->extract_data_dict();
    }
    if(early_stop) {
        ROS_WARN("%s Early stop", LOCAL_AGENT.c_str());
        return 0;
    }
    
    timer.Start();
    bool shape_encode_ret = loop_detector->encode_concat_sgs(ref_agent_name,
                                        ref_graph->get_const_nodes().size(), ref_data_dict,
                                        src_graph->get_const_nodes().size(), src_data_dict, fused);
    timer.Stop();
    std::cout<<"Encode shape takes "<<std::fixed<<std::setprecision(3)
            <<timer.GetDurationInMillisecond()<<" ms\n";

    // Hierachical matching
    std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
    std::vector<float> match_scores;
    std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;
    std::vector<float> corr_scores_vec;

    int M; // number of matched nodes
    int C=0; // number of matched points
    std::vector <NodePair> pruned_match_pairs;
    std::vector<float> pruned_match_scores;

    M = loop_detector->match_nodes(ref_agent_name,match_pairs, match_scores, fused);
    std::vector<bool> pruned_true_masks(M, false);
    ROS_WARN("Matched nodes: %d", M);
    if(prune_instances && M>0){        
        Registration::pruneInsOutliers(sg_config->reg,
                                        src_graph->get_const_nodes(),
                                        ref_graph->get_const_nodes(),
                                        match_pairs,
                                        pruned_true_masks);
    }
    else{
        pruned_true_masks = std::vector<bool>(M, true);
    }

    pruned_match_pairs = fmfusion::utility::update_masked_vec(match_pairs, pruned_true_masks);
    pruned_match_scores = fmfusion::utility::update_masked_vec(match_scores, pruned_true_masks);
    ROS_WARN("Keep %d matched nodes after prune", pruned_match_pairs.size());

    if(pruned_match_pairs.size()>0 && shape_encode_ret){ // dense match  
        C = loop_detector->match_instance_points(ref_agent_name,
                                                pruned_match_pairs, 
                                                corr_src_points, 
                                                corr_ref_points, 
                                                corr_scores_vec);
        ROS_WARN("Matched points: %d", C);
    }

    // Registration
    fmfusion::O3d_Cloud_Ptr src_cloud_ptr = src_mapping->export_global_pcd(true, 0.05);
    fmfusion::O3d_Cloud_Ptr ref_cloud_ptr = ref_mapping->export_global_pcd(true, 0.05);
    Eigen::Matrix4d pred_pose;

    g3reg::Config config;
    config.set_noise_bounds({0.2, 0.3});
    config.tf_solver  = "quatro";
    config.verify_mtd = "plane_based";
    G3RegAPI g3reg(config);

    std::cout<<"Estimate pose with "<<pruned_match_pairs.size()<<" nodes and"
                <<corr_src_points.size()<<" points\n";
    g3reg.estimate_pose(src_graph->get_const_nodes(),
                        ref_graph->get_const_nodes(),
                        pruned_match_pairs,
                        corr_scores_vec,
                        corr_src_points,
                        corr_ref_points,
                        src_cloud_ptr,
                        ref_cloud_ptr,
                        pred_pose);

    std::cout<<"trying icp\n";
    timer.Start();
    if(icp_refine){
        if(!ref_cloud_ptr->HasNormals()) ref_cloud_ptr->EstimateNormals();
        pred_pose = g3reg.icp_refine(src_cloud_ptr, ref_cloud_ptr, pred_pose);  
    }      
    timer.Stop();
    std::cout<<"ICP refine takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";

    // Export
    std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> pred_instances;
    std::vector<bool> pred_masks;
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
    {
        fmfusion::IO::extract_match_instances(
            pruned_match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), pred_instances);
        fmfusion::IO::extract_instance_correspondences(
            src_graph->get_const_nodes(), ref_graph->get_const_nodes(), pruned_match_pairs, pruned_match_scores, src_centroids, ref_centroids);

        if(output_folder.size()>1){
            std::cout<<"output size: "<<output_folder.size()<<std::endl;
            fmfusion::IO::save_match_results(pred_pose,
                                            pred_instances,
                                            pruned_match_scores,
                                            output_folder+"/"+src_name+"-"+ref_name+".txt");
            open3d::io::WritePointCloudToPLY(output_folder + "/" + src_name + ".ply", *src_cloud_ptr, {});
        
            if (C > 0) {
                fmfusion::O3d_Cloud_Ptr corr_src_pcd = std::make_shared<fmfusion::O3d_Cloud>(corr_src_points);
                fmfusion::O3d_Cloud_Ptr corr_ref_pcd = std::make_shared<fmfusion::O3d_Cloud>(corr_ref_points);
                open3d::io::WritePointCloudToPLY(output_folder + "/" + pair_name + "_csrc.ply", *corr_src_pcd, {});
                open3d::io::WritePointCloudToPLY(output_folder + "/" + pair_name + "_cref.ply", *corr_ref_pcd, {});
            }
        }
    }

    //
    if(gt_file.size()>0){
        int count_true = fmfusion::maks_true_instance(gt_file, pred_instances, pred_masks);
        std::cout<<"True instance match: "<<count_true<<"/"<<pred_instances.size()<<std::endl;
    }

    if(visualization>0){
        ROS_WARN("run visualization");
        if(test_communication)
            Visualization::render_point_cloud(ref_graph->extract_global_cloud(), viz.ref_graph, REMOTE_AGENT);
        else
            Visualization::render_point_cloud(ref_mapping->export_global_pcd(true), viz.ref_graph,REMOTE_AGENT);
        ros::Duration(1.0).sleep();
        Visualization::instance_centroids(ref_graph->get_centroids(),viz.ref_centroids,REMOTE_AGENT,viz.param.centroid_size,viz.param.centroid_color);

        Visualization::render_point_cloud(src_mapping->export_global_pcd(true), viz.src_graph, LOCAL_AGENT);
        Visualization::instance_centroids(src_graph->get_centroids(),viz.src_centroids,LOCAL_AGENT,viz.param.centroid_size,viz.param.centroid_color);
        Visualization::inter_graph_edges(src_graph->get_centroids(), src_graph->get_edges(), viz.src_edges, viz.param.edge_width, viz.param.edge_color , LOCAL_AGENT);

        Visualization::correspondences(src_centroids, ref_centroids, viz.instance_match,LOCAL_AGENT, pred_masks, viz.t_local_remote[ref_agent_name]);
        Visualization::correspondences(corr_src_points, corr_ref_points, viz.point_match, LOCAL_AGENT, std::vector<bool> {}, viz.t_local_remote[ref_agent_name]);

        if(viz.src_map_aligned.getNumSubscribers()>0){
            ros::Duration(1.0).sleep();
            auto global_src_cloud_ptr = src_mapping->export_global_pcd();
            global_src_cloud_ptr->Transform(pred_pose);
            global_src_cloud_ptr->PaintUniformColor({0.0,0.707,0.707});
            Visualization::render_point_cloud(global_src_cloud_ptr, viz.src_map_aligned, REMOTE_AGENT);
            ROS_INFO("Publish aligned source map");
        }
    }

    ros::spinOnce();

    ros::Duration(1.0).sleep();
    ROS_WARN("Shutting down ros ...");
    ros::shutdown();

    return 0;

}