#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>

#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/g3reg_api.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"
#include "sgloop/LoopDetector.h"

#include "registration/Prune.h"
#include "communication/Communication.h"
#include "Visualization.h"

typedef fmfusion::IO::RGBDFrameDirs RGBDFrameDirs;

struct Publishers{
    ros::Publisher rgb_camera;
}mypub;

void save_config(const fmfusion::Config &config, std::string out_folder)
{
    using namespace open3d::utility::filesystem;
    if(!DirectoryExists(out_folder)) MakeDirectory(out_folder);

    std::ofstream out_file(out_folder+"/config.txt");
    out_file<<fmfusion::utility::config_to_message(config);
    out_file.close();
}

void rgbd_callback(const ros::TimerEvent &event)
{
    std::cout<<"aa \n";
    ros::Duration(0.2).sleep();
}

int main(int argc, char **argv)
{
    using namespace fmfusion;
    using namespace open3d::utility;
    using namespace open3d::io;

    ros::init(argc, argv, "LoopNode");
    ros::NodeHandle n;
    ros::NodeHandle nh_private("~");

    // Settings
    std::string config_file, weights_folder;
    std::string ref_scene_dir, root_dir;
    std::string LOCAL_AGENT, REMOTE_AGENT;

    mypub.rgb_camera = nh_private.advertise<sensor_msgs::Image>("rgb/image_raw", 1);

    assert(nh_private.getParam("cfg_file", config_file));
    bool set_wegith_folder = nh_private.getParam("weights_folder", weights_folder);
    bool set_src_scene = nh_private.getParam("active_sequence_dir", root_dir);
    int frame_gap = nh_private.param("frame_gap", 1);
    bool set_local_agent = nh_private.getParam("local_agent", LOCAL_AGENT);
    bool set_remote_agent = nh_private.getParam("remote_agent", REMOTE_AGENT);
    assert(set_wegith_folder && set_src_scene && set_local_agent);
    bool set_ref_scene = nh_private.getParam("ref_scene_dir", ref_scene_dir);

    std::string prediction_folder = nh_private.param("prediction_folder", std::string("prediction_no_augment"));
    std::string output_folder = nh_private.param("output_folder", std::string(""));
    int visualization = nh_private.param("visualization", 0);
    float loop_duration = nh_private.param("loop_duration", 20.0);
    int mode = nh_private.param("mode", 0); // mode 0: load ref map, 1: real-time subscribe ref map
    bool debug_mode = nh_private.param("debug_mode", false);

    SetVerbosityLevel((VerbosityLevel)5);
    LogInfo("Read configuration from {:s}",config_file);
    LogInfo("Read RGBD sequence from {:s}", root_dir);
    std::string sequence_name = *filesystem::GetPathComponents(root_dir).rbegin();
    std::string ref_name = *filesystem::GetPathComponents(ref_scene_dir).rbegin();

    // loop detection result
    std::string loop_result_dir = output_folder+"/"+sequence_name+"/"+ref_name;
    if(!filesystem::DirectoryExists(loop_result_dir)) filesystem::MakeDirectory(loop_result_dir);

    //
    auto global_config = utility::create_scene_graph_config(config_file, true);
    save_config(*global_config, output_folder+"/"+sequence_name);

    // Load frames information
    std::vector<RGBDFrameDirs> rgbd_table;
    std::vector<Eigen::Matrix4d> pose_table;
    bool read_ret = fmfusion::IO::construct_preset_frame_table(root_dir,"data_association.txt","trajectory.log",rgbd_table,pose_table);
    if(!read_ret || global_config==nullptr) {
        return 0;
    }

    // Loop detector
    auto loop_detector = std::make_shared<LoopDetector>(LoopDetector(
                                                                global_config->loop_detector,
                                                                global_config->shape_encoder, 
                                                                global_config->sgnet,
                                                                weights_folder));

    // Reference map
    open3d::utility::Timer timer;
    std::vector<InstanceId> ref_names;
    std::vector<InstancePtr> ref_instances;    
    std::shared_ptr<SemanticMapping> ref_mapping = std::make_shared<SemanticMapping>(
                                        SemanticMapping(global_config->mapping_cfg,global_config->instance_cfg));
    std::shared_ptr<Graph> ref_graph = std::make_shared<Graph>(global_config->graph);
    DataDict ref_data_dict;
    std::shared_ptr<SgCom::Communication> comm;

    if(mode==0){   // Load
        timer.Start();
        ref_mapping->load(ref_scene_dir);
        ref_mapping->extract_bounding_boxes();
        ref_mapping->export_instances(ref_names,ref_instances);
        
        ref_graph->initialize(ref_instances);
        ref_graph->construct_edges();
        ref_graph->construct_triplets();
        ref_data_dict = ref_graph->extract_data_dict();    
        timer.Stop();
        std::cout<<"Extract ref scene takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";

        loop_detector->encode_ref_scene_graph(ref_graph->get_const_nodes(),DataDict {});
        ros::Duration(1.0).sleep();
    }
    else{ // Communication server
        std::vector<std::string> remote_agents = {REMOTE_AGENT};
        comm = std::make_shared<SgCom::Communication>(n,nh_private, LOCAL_AGENT, remote_agents);
        ROS_WARN("Init communication server for %s", LOCAL_AGENT.c_str());
    }

    // Local mapping module
    fmfusion::SemanticMapping semantic_mapping(global_config->mapping_cfg, global_config->instance_cfg);
    auto src_graph = std::make_shared<Graph>(global_config->graph);
    open3d::geometry::Image depth, color;
    int prev_frame_id = -100;
    int loop_frame_id = -100;

    // Viz
    Visualization::Visualizer viz(n,nh_private);
    if(mode==0)
    {
        ros::Duration(0.5).sleep();
        Visualization::instance_centroids(ref_graph->get_centroids(),viz.ref_centroids,REMOTE_AGENT,viz.param.centroid_size);
        ros::Duration(1.0).sleep();
        Visualization::render_point_cloud(ref_mapping->export_global_pcd(true,0.05), viz.ref_graph,REMOTE_AGENT);
        ros::Duration(0.5).sleep();
        ROS_WARN("Load and render reconstrcuted reference scene graph");
    }
    int loop_count = 0;

    // Running sequence
    fmfusion::TicTocSequence tic_toc_seq("# Load Mapping SgCreate SgCoarse Boadcast Prune Viz");
    ROS_WARN("[%s] Read to run sequence", LOCAL_AGENT.c_str());
    ros::Duration(1.0).sleep();

    for(int k=0;k<rgbd_table.size();k++){
        RGBDFrameDirs frame_dirs = rgbd_table[k];
        std::string frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
        frame_name = frame_name.substr(0,frame_name.find_last_of("."));
        int seq_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));

        if((seq_id-prev_frame_id)<frame_gap) continue;
        tic_toc_seq.tic();
        LogInfo("Processing frame {:s} ...", frame_name);

        ReadImage(frame_dirs.second, depth);
        ReadImage(frame_dirs.first, color);

        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, global_config->mapping_cfg.depth_scale, global_config->mapping_cfg.depth_max, false);
        
        std::vector<DetectionPtr> detections;
        bool loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, frame_name, 
                                                        global_config->mapping_cfg, global_config->instance_cfg.intrinsic.width_, global_config->instance_cfg.intrinsic.height_,
                                                        detections);
        tic_toc_seq.toc();
        if(loaded){
            semantic_mapping.integrate(seq_id,rgbd, pose_table[k], detections);
            prev_frame_id = seq_id;
        }
        tic_toc_seq.toc();

        if(seq_id-loop_frame_id > loop_duration){
            std::vector<fmfusion::InstanceId> valid_names;
            std::vector<InstancePtr> valid_instances;
            semantic_mapping.export_instances(valid_names, valid_instances);

            if(valid_names.size()>global_config->loop_detector.lcd_nodes){ // Local ready to detect loop
                std::vector<NodePair> match_pairs, pruned_match_pairs;
                std::vector<float> match_scores, pruned_match_scores;
                std::vector<Eigen::Vector3d> src_centroids, ref_centroids;

                LogWarning("[LCD] Detecting loop closure with {:d} valid instances", valid_names.size());
                std::cout<<"loop id: "<<loop_count<<std::endl;

                // Explicit local
                src_graph->clear();
                src_graph->initialize(valid_instances);
                src_graph->construct_edges();
                src_graph->construct_triplets();
                fmfusion::DataDict src_node_coarse_dict = src_graph->extract_data_dict(true);
                tic_toc_seq.toc();

                // Implicit local
                loop_detector->encode_src_scene_graph(src_graph->get_const_nodes(), DataDict {});
                tic_toc_seq.toc();

                // 
                int Ns, Ds;
                int Nr;
                std::vector<std::vector<float>> src_node_feats_vec;
                if(mode==1){ // Broadcast and update subscribed ref map
                    loop_detector->get_active_node_feats(src_node_feats_vec, Ns, Ds);
                    comm->broadcast_coarse_graph(seq_id,
                                                src_node_coarse_dict.instances,
                                                src_node_coarse_dict.centroids,
                                                Ns, Ds,
                                                src_node_feats_vec);

                    const SgCom::AgentDataDict received_data = comm->get_remote_agent_data(REMOTE_AGENT);
                    if(received_data.frame_id>=0){
                        bool exp_update_flag = ref_graph->subscribe_coarse_nodes(received_data.received_timestamp,
                                                                                received_data.instances,
                                                                                received_data.centroids);
                        bool imp_update_flag = loop_detector->subscribe_ref_coarse_features(received_data.received_timestamp,
                                                                                        received_data.features_vec,
                                                                                        torch::empty({0,0}));
                        Nr = received_data.N;
                        if(debug_mode){ // visualize the updated ref graph
                            std::cout<<"Nr: "<<Nr
                                    <<" ref graph nodes: "<<ref_graph->get_const_nodes().size()
                                    <<" ref features: "<<received_data.features_vec.size()<<std::endl;
                            assert(loop_detector->get_ref_feats_number()==ref_graph->get_const_nodes().size());                        
                            std::array<float,3> highlight_color = {1.0,0.0,0.0};
                            Visualization::instance_centroids(ref_graph->get_centroids(),viz.ref_centroids,REMOTE_AGENT,viz.param.centroid_size,highlight_color);
                            // torch::Tensor ref_feats = loop_detector->get_ref_node_feats();
                            // torch::Tensor src_feats = loop_detector->get_active_node_feats();
                            // o3d_utility::LogWarning("debug info \n");
                            // std::cout<<"src instances: "<< src_graph->print_nodes_names()<<"\n";
                            // std::cout<<"ref instances: "<< ref_graph->print_nodes_names()<<"\n";
                            // if(Ns==Nr){
                            //     int src_nan_sum = torch::isnan(src_feats).sum().item<int>();
                            //     int ref_nan_sum = torch::isnan(ref_feats).sum().item<int>();
                            //     std::cout<<"src nan sum: "<<src_nan_sum
                            //                 <<" ref nan sum: "<<ref_nan_sum<<std::endl;
                            //     bool verified = torch::allclose(ref_feats, src_feats, 1e-5);
                            //     if(verified) o3d_utility::LogWarning("Verified ref and src features");
                            //     else {
                            //         o3d_utility::LogWarning("Ref and src features mismatch");
                            //         // ros::shutdown();
                            //         // break;
                            //     }
                            // }

                        }

                    }
                    else Nr = 0;
                }
                else Nr = ref_data_dict.instances.size();

                // Loop closure
                int M=0;
                if(Nr>global_config->loop_detector.lcd_nodes){
                    M = loop_detector->match_nodes(match_pairs, match_scores, false);
                    std::cout<<"Find "<<M<<" matched nodes\n";
                }

                tic_toc_seq.toc();

                //
                if(M>0){
                    std::vector<bool> pruned_true_masks(M, false);
                    std::vector<NodePair> pruned_match_pairs;
                    std::vector<float> pruned_match_scores;
                    std::vector <std::pair<fmfusion::InstanceId, fmfusion::InstanceId>> match_instances;

                    Registration::pruneInsOutliers(global_config->reg, 
                                            src_graph->get_const_nodes(), ref_graph->get_const_nodes(), 
                                            match_pairs, pruned_true_masks);
                    pruned_match_pairs = fmfusion::utility::update_masked_vec(match_pairs, pruned_true_masks);
                    pruned_match_scores = fmfusion::utility::update_masked_vec(match_scores, pruned_true_masks);
                    std::cout<<"Keep "<<pruned_match_pairs.size()<<" consistent matched nodes\n";
                    tic_toc_seq.toc();

                    fmfusion::IO::extract_match_instances(pruned_match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_instances);
                    fmfusion::IO::extract_instance_correspondences(src_graph->get_const_nodes(), ref_graph->get_const_nodes(), 
                                                                pruned_match_pairs, pruned_match_scores, src_centroids, ref_centroids);                    
                    Visualization::correspondences(src_centroids, ref_centroids, viz.instance_match,LOCAL_AGENT,{},viz.local_frame_offset);
                
                    Eigen::Matrix4d pred_pose;
                    pred_pose.setIdentity();
                    std::cout<<"Write "<<match_instances.size()<<" matched instances\n";
                    fmfusion::IO::save_match_results(pred_pose, match_instances, pruned_match_scores, loop_result_dir+"/"+frame_name+".txt");
                }
                else tic_toc_seq.fill_zeros();

                loop_count ++;
            }
            else{
                tic_toc_seq.fill_zeros(4);
            }

            //
            Visualization::instance_centroids(semantic_mapping.export_instance_centroids(),viz.src_centroids,LOCAL_AGENT,viz.param.centroid_size);
            Visualization::render_point_cloud(semantic_mapping.export_global_pcd(true,0.05), viz.src_graph, LOCAL_AGENT);
            tic_toc_seq.toc();
            loop_frame_id = seq_id;
        }   


        if(viz.rgb_image.getNumSubscribers()>0){
            auto rgb_cv = std::make_shared<cv::Mat>(color.height_,color.width_,CV_8UC3);
            memcpy(rgb_cv->data,color.data_.data(),color.data_.size()*sizeof(uint8_t));            
            Visualization::render_image(*rgb_cv, viz.rgb_image, LOCAL_AGENT);
        }
        Visualization::render_camera_pose(pose_table[k], viz.camera_pose, LOCAL_AGENT, seq_id);
        Visualization::render_path(pose_table[k], viz.path_msg, viz.path, LOCAL_AGENT, seq_id);

        ros::spinOnce();
    }
    LogWarning("Finished sequence");
    ros::shutdown();

    // Post-process
    semantic_mapping.extract_point_cloud();
    semantic_mapping.merge_overlap_instances();
    semantic_mapping.merge_overlap_structural_instances();
    semantic_mapping.extract_bounding_boxes();

    // Save
    semantic_mapping.Save(output_folder+"/"+sequence_name);
    tic_toc_seq.export_data(output_folder+"/"+sequence_name+"/timing.txt");
    if(mode==1) comm->write_logs(output_folder+"/"+sequence_name);
    LogInfo("Save sequence to {:s}", output_folder+"/"+sequence_name);
    LogInfo("Loop results saved to {:s}", loop_result_dir);

    // ros::spin();
    return 0;
}