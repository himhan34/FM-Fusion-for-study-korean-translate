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
#include "registration/robustPoseAvg.h"
#include "communication/Communication.h"
#include "Visualization.h"
#include "SlidingWindow.h"

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

struct DenseMatch{
    int last_frame_id = 0;
    int coarse_loop_number = 0;
};

struct DenseMatchConfig{
    int min_loops = 3;
    int min_frame_gap = 200;
    float broadcast_sleep_time; // sleep time after broadcast dense graph
};

/// @brief Take the ref agent info form the communication node. 
///        And update ref graph and implicit features in loop detector accordingly.
bool remote_sg_update(const std::string &target_agent, 
                    const SgCom::AgentDataDict &received_data,
                    std::shared_ptr<fmfusion::LoopDetector> &loop_detector,
                    std::shared_ptr<fmfusion::Graph> &target_graph,
                    fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr)
{
    if(received_data.frame_id>0 &&received_data.N>=0){ // subscribtion update
        bool imp_update_flag = loop_detector->subscribe_ref_coarse_features(target_agent,
                                                                        received_data.received_timestamp,
                                                                        received_data.features_vec,
                                                                        torch::empty({0,0}));                        
        int update_nodes_count  = target_graph->subscribe_coarse_nodes(received_data.received_timestamp,
                                                                received_data.nodes,
                                                                received_data.instances,
                                                                received_data.centroids);
        int update_points_count = 0;
        if(update_nodes_count>0 && received_data.X>0){
            update_points_count = target_graph->subscribde_dense_points(received_data.received_timestamp,
                                                                        received_data.xyz,
                                                                        received_data.labels);
            ref_cloud_ptr = std::make_shared<open3d::geometry::PointCloud>(received_data.xyz);
        }
        if(update_nodes_count>0){
            std::cout<<"update "<< update_nodes_count<<" ref nodes and " 
                                << update_points_count<<" ref points from "
                                <<target_agent<<"\n";
            // return true;
        }
        // else return false;
        return true;
    }
    else return false;

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
    std::string LOCAL_AGENT;// REMOTE_AGENT;
    std::string secondAgent, thirdAgent;
    std::string secondAgentScene, thirdAgentScene;

    mypub.rgb_camera = nh_private.advertise<sensor_msgs::Image>("rgb/image_raw", 1);

    assert(nh_private.getParam("cfg_file", config_file));
    bool set_wegith_folder = nh_private.getParam("weights_folder", weights_folder);
    bool set_src_scene = nh_private.getParam("active_sequence_dir", root_dir);
    int frame_gap = nh_private.param("frame_gap", 1);
    bool set_local_agent = nh_private.getParam("local_agent", LOCAL_AGENT);
    // bool set_remote_agent = nh_private.getParam("remote_agent", REMOTE_AGENT);
    bool set_2nd_agent = nh_private.getParam("second_agent", secondAgent);
    bool set_3rd_agent = nh_private.getParam("third_agent", thirdAgent);
    assert(set_wegith_folder && set_src_scene && set_local_agent);
    bool set_2nd_agent_scene = nh_private.getParam("second_agent_scene", secondAgentScene);
    bool set_3rd_agent_scene = nh_private.getParam("third_agent_scene",  thirdAgentScene);
    // bool set_ref_scene = nh_private.getParam("ref_scene_dir", ref_scene_dir);

    std::string prediction_folder = nh_private.param("prediction_folder", std::string("prediction_no_augment"));
    std::string output_folder = nh_private.param("output_folder", std::string(""));
    int visualization = nh_private.param("visualization", 0);
    float loop_duration = nh_private.param("loop_duration", 20.0);
    bool prune_instances = nh_private.param("prune_instances", true);
    bool debug_mode = nh_private.param("debug_mode", false);
    bool icp_refine = nh_private.param("icp_refine",false);
    int pose_average_window = nh_private.param("pose_average_size",5);
    bool save_corr = nh_private.param("save_corr",true);
    int o3d_verbose_level = nh_private.param("o3d_verbose_level", 2);
    float sliding_widow_translation = nh_private.param("sliding_widow_translation", 100.0);

    // Remote agents    
    assert(set_2nd_agent && set_2nd_agent_scene);
    std::vector<std::string> remote_agents = {secondAgent};
    std::vector<std::string> remote_agents_scenes = {secondAgentScene};
    std::vector<int> remote_agents_loop_frames = {0};
    std::vector<std::string> loop_results_dirs;

    if(set_3rd_agent){
        remote_agents.push_back(thirdAgent);
        remote_agents_scenes.push_back(thirdAgentScene);
        remote_agents_loop_frames.push_back(0);
    }
    std::vector<DenseMatch> dense_match(remote_agents.size());

    // outputs
    std::string sequence_name = *filesystem::GetPathComponents(root_dir).rbegin();
    if(!filesystem::DirectoryExists(output_folder)) filesystem::MakeDirectory(output_folder);
    for(auto scene_name:remote_agents_scenes){
        std::string loop_result_dir = output_folder+"/"+sequence_name+"/"+scene_name;
        if(!filesystem::DirectoryExists(loop_result_dir)) filesystem::MakeDirectoryHierarchy(loop_result_dir);
        loop_results_dirs.push_back(loop_result_dir);
    }
    
    // Configs
    fmfusion::Config *global_config;
    DenseMatchConfig dense_m_config;
    {
        global_config = utility::create_scene_graph_config(config_file, true);
        save_config(*global_config, output_folder+"/"+sequence_name);
        dense_m_config.min_loops = nh_private.param("dense_m_loops", 3);
        dense_m_config.min_frame_gap = nh_private.param("dense_m_frame_gap", 200);
        dense_m_config.broadcast_sleep_time = nh_private.param("broadcast_sleep_time", 1.5);
        SetVerbosityLevel((VerbosityLevel)o3d_verbose_level);
    }

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
                                                                weights_folder,
                                                                0,
                                                                remote_agents));



    //
    std::shared_ptr<SgCom::Communication> comm;
    comm = std::make_shared<SgCom::Communication>(n,nh_private, LOCAL_AGENT, remote_agents);
    ROS_WARN("Init communication server for %s", LOCAL_AGENT.c_str());
    
    // Local mapping and graph
    fmfusion::SemanticMapping semantic_mapping(global_config->mapping_cfg, global_config->instance_cfg);
    std::vector<fmfusion::InstanceId> valid_names;
    std::vector<InstancePtr> valid_instances;
    auto src_graph = std::make_shared<Graph>(global_config->graph);
    int prev_frame_id = -100;
    int broadcast_frame_id = -100;

    // G3Reg
    g3reg::Config config;
    config.set_noise_bounds({0.2, 0.3});
    config.tf_solver  = "quatro";
    config.verify_mtd = "plane_based";
    G3RegAPI g3reg(config);

    // Viz
    Visualization::Visualizer viz(n,nh_private);

    // Running sequence
    fmfusion::TimingSequence timing_seq("#FrameID: Load Mapping; SgCreate SgCoarse Pub&Sub ShapeEncoder PointMatch; Match Reg Viz");
    robot_utils::TicToc tic_toc;
    robot_utils::TicToc tictoc_bk;

    // Reference map
    std::vector<GraphPtr> remote_graphs;
    std::vector<std::vector<gtsam::Pose3>> remote_pred_poses;
    open3d::utility::Timer timer;
    for(int k=0;k<remote_agents.size();k++){
        remote_graphs.push_back(std::make_shared<Graph>(global_config->graph));
        remote_pred_poses.push_back(std::vector<gtsam::Pose3>());
    }

    // Active scene graph
    open3d::geometry::Image depth, color;
    O3d_Cloud_Ptr cur_active_instance_pcd;
    fmfusion::DataDict src_data_dict;
    int Ns;
    int loop_count = 0;
    bool global_coarse_mode = true; // If coarse loop detected, set to false;
    sgloop_ros::SlidingWindow sliding_window(sliding_widow_translation);

    ROS_WARN("[%s] Read to run sequence", LOCAL_AGENT.c_str());
    ros::Duration(1.0).sleep();

    for(int k=0;k<rgbd_table.size();k++){
        RGBDFrameDirs frame_dirs = rgbd_table[k];
        std::string frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
        frame_name = frame_name.substr(0,frame_name.find_last_of("."));
        int seq_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));
        if((seq_id-prev_frame_id)<frame_gap) continue;
        timing_seq.create_frame(seq_id);
        std::cout<<"["<< LOCAL_AGENT<<"] " <<"Processing frame "<<seq_id<<"\n";

        bool loaded;
        std::vector<DetectionPtr> detections;
        std::shared_ptr<open3d::geometry::RGBDImage> rgbd;

        { // load RGB-D, detections
            tic_toc.tic();
            ReadImage(frame_dirs.second, depth);
            ReadImage(frame_dirs.first, color);
            rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
                color, depth, global_config->mapping_cfg.depth_scale, global_config->mapping_cfg.depth_max, false);
            
            loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, frame_name, 
                                                            global_config->mapping_cfg, global_config->instance_cfg.intrinsic.width_, global_config->instance_cfg.intrinsic.height_,
                                                            detections);
            timing_seq.record(tic_toc.toc()); 
        }

        if(loaded){
            semantic_mapping.integrate(seq_id,rgbd, pose_table[k], detections);
            prev_frame_id = seq_id;
        }
        timing_seq.record(tic_toc.toc()); // mapping

        if(seq_id-broadcast_frame_id > loop_duration){
            valid_names.clear();
            valid_instances.clear();
            semantic_mapping.export_instances(valid_names, valid_instances, 
                                                sliding_window.get_window_start_frame());
            cur_active_instance_pcd = semantic_mapping.export_global_pcd(true,0.05);

            // Local encode and broadcast
            if(valid_names.size()>global_config->loop_detector.lcd_nodes){ // Local ready to detect loop
                ROS_WARN("** [%s] loop id: %d, active nodes: %d **", 
                        LOCAL_AGENT.c_str(), loop_count, valid_names.size());

                // Explicit local
                Ns = 0;
                src_graph->clear();
                src_graph->initialize(valid_instances);
                src_graph->construct_edges();
                src_graph->construct_triplets();

                if(comm->get_pub_dense_msg())
                { // Been called. Extract dense msg to broadcast. 
                    global_coarse_mode = false;
                    comm->reset_pub_dense_msg();
                    comm->pub_dense_frame_id = seq_id;
                    ROS_WARN("%s been called and set to dense mode", LOCAL_AGENT.c_str());
                }

                src_data_dict.clear();
                src_data_dict = src_graph->extract_data_dict(global_coarse_mode);
                timing_seq.record(tic_toc.toc());// construct sg

                // Implicit local
                tictoc_bk.tic();
                loop_detector->encode_src_scene_graph(src_graph->get_const_nodes());
                ROS_WARN("Encode %d nodes takes %.3f ms", src_graph->get_const_nodes().size(), tictoc_bk.toc());
                timing_seq.record(tic_toc.toc());// coarse encode

                // Comm
                std::vector<std::vector<float>> src_node_feats_vec;
                tictoc_bk.tic();

                { // publish master agent sg
                    int Ds;
                    loop_detector->get_active_node_feats(src_node_feats_vec, Ns, Ds);
                    bool broadcast_ret = comm->broadcast_dense_graph(seq_id,
                                                src_data_dict.nodes,
                                                src_data_dict.instances,
                                                src_data_dict.centroids,
                                                src_node_feats_vec,
                                                src_data_dict.xyz,
                                                src_data_dict.labels);
                    // reset to coarse mode after publish dense
                    if(broadcast_ret && src_data_dict.xyz.size()>10) {
                        global_coarse_mode = true;
                        ROS_WARN("%s Broadcast dense graph", LOCAL_AGENT.c_str());
                        if(remote_graphs[0]->get_timestamp()<0.0) // never received any remote graph
                            ros::Duration(dense_m_config.broadcast_sleep_time).sleep();
                    }
                }
                
            }
            
        }   

        //todo: iter over remote agents
        // Update one remote agent and check loop
        if(valid_names.size()>global_config->loop_detector.lcd_nodes){
            int Nr;
            int &prev_loop_frame_id = remote_agents_loop_frames[0];
            GraphPtr target_graph = remote_graphs[0];
            std::string target_agent = remote_agents[0];
            std::string loop_result_dir = loop_results_dirs[0];
            DenseMatch &target_dense_m = dense_match[0];
            std::vector<gtsam::Pose3> &poses = remote_pred_poses[0];

            const SgCom::AgentDataDict &received_data = comm->get_remote_agent_data(target_agent);
            O3d_Cloud_Ptr ref_cloud_ptr(new O3d_Cloud());
            bool sg_updated = remote_sg_update(target_agent, received_data, loop_detector, target_graph, ref_cloud_ptr);
            timing_seq.record(tic_toc.toc());// Sub + pub

            if(sg_updated && (seq_id - prev_loop_frame_id)>loop_duration){
                { // Shape encode
                    DataDict ref_data_dict = target_graph->extract_data_dict();
                    Nr = ref_data_dict.instances.size();
                    bool shape_ret = loop_detector->encode_concat_sgs(target_agent,
                                                    Nr, ref_data_dict, 
                                                    Ns, src_data_dict, true);
                    
                    if(shape_ret) { // reset
                        global_coarse_mode = true; // src graph back to coarse mode
                    }
                    
                }
                timing_seq.record(tic_toc.toc());// shape encode

                // Loop closure
                std::vector<NodePair> match_pairs, pruned_match_pairs;
                std::vector<float> match_scores, pruned_match_scores;
                int M=0;
                if(Nr>global_config->loop_detector.lcd_nodes){ // coarse match
                    M = loop_detector->match_nodes(remote_agents[0],match_pairs, match_scores, true);
                    std::cout<<"Find "<<M<<" matched nodes\n";
                }
                timing_seq.record(tic_toc.toc());// match

                // Points Match
                int C = 0;
                std::vector<bool> pruned_true_masks(M, false);
                std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
                std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;
                std::vector<float> corr_scores_vec;
                fmfusion::O3d_Cloud_Ptr src_cloud_ptr = cur_active_instance_pcd;

                if(M>global_config->loop_detector.recall_nodes){ // prune + dense match
                    tictoc_bk.tic();
                    if(prune_instances)
                        Registration::pruneInsOutliers(global_config->reg, 
                                                src_graph->get_const_nodes(), target_graph->get_const_nodes(), 
                                                match_pairs, pruned_true_masks);
                    else{
                        pruned_true_masks = std::vector<bool>(M, true);
                    }
                    pruned_match_pairs = fmfusion::utility::update_masked_vec(match_pairs, pruned_true_masks);
                    pruned_match_scores = fmfusion::utility::update_masked_vec(match_scores, pruned_true_masks);
                    M = pruned_match_pairs.size();
                    std::cout<<"Keep "<<M<<" consistent matched nodes\n";
                    if(M>global_config->loop_detector.recall_nodes)
                        target_dense_m.coarse_loop_number++;

                    C = loop_detector->match_instance_points(target_agent,
                                                            pruned_match_pairs, 
                                                            corr_src_points, 
                                                            corr_ref_points, 
                                                            corr_scores_vec);
                    std::cout<<"Find "<<C<<" matched points\n";
                }
                timing_seq.record(tic_toc.toc());// Point match

                // Pose Estimation
                Eigen::Matrix4d pred_pose;
                pred_pose.setIdentity();
                tic_toc.tic();
                if(M>global_config->loop_detector.recall_nodes){
                    tictoc_bk.tic();
                    g3reg.estimate_pose(src_graph->get_const_nodes(),
                                        target_graph->get_const_nodes(),
                                        pruned_match_pairs,
                                        corr_scores_vec,
                                        corr_src_points,
                                        corr_ref_points,
                                        src_cloud_ptr,
                                        ref_cloud_ptr,
                                        pred_pose);
                    // std::cout<<"Pose estimation takes "<<tictoc_bk.toc()<<" ms\n";

                    if(icp_refine && ref_cloud_ptr->HasPoints()){
                        if(!ref_cloud_ptr->HasNormals()) ref_cloud_ptr->EstimateNormals();
                        pred_pose = g3reg.icp_refine(src_cloud_ptr, ref_cloud_ptr, pred_pose);
                        ROS_WARN("ICP refine pose");
                    } 

                    if(pose_average_window>1){
                        tictoc_bk.tic();
                        gtsam::Pose3 cur_pred_pose(gtsam::Rot3(pred_pose.block<3,3>(0,0)),
                                                    gtsam::Point3(pred_pose.block<3,1>(0,3)));
                        poses.push_back(cur_pred_pose);
                        if(poses.size()>pose_average_window)
                            poses.erase(poses.begin());
                        
                        gtsam::Pose3 robust_pred_pose = registration::gncRobustPoseAveraging(poses);
                        pred_pose = robust_pred_pose.matrix();
                        std::cout<<"Pose averaging "<<poses.size()<<" poses, "
                                << " takes "<<tictoc_bk.toc()<<" ms\n";

                    }                    
                }

                if(target_dense_m.coarse_loop_number>dense_m_config.min_loops &&
                    (seq_id - target_dense_m.last_frame_id)>dense_m_config.min_frame_gap){
                    // Send ONE request message.
                    comm->send_request_dense(target_agent);
                    global_coarse_mode = false;
                    target_dense_m.coarse_loop_number = 0;
                    target_dense_m.last_frame_id = seq_id;
                    ROS_WARN("%s Request dense message",LOCAL_AGENT.c_str());
                }

                timing_seq.record(tic_toc.toc());// Pose

                // I/O
                std::vector <std::pair<fmfusion::InstanceId, fmfusion::InstanceId>> match_instances;
                if(M>global_config->loop_detector.recall_nodes){ //IO
                    fmfusion::IO::extract_instance_correspondences(src_graph->get_const_nodes(), target_graph->get_const_nodes(), 
                                                                pruned_match_pairs, pruned_match_scores, src_centroids, ref_centroids);                
                    fmfusion::IO::extract_match_instances(pruned_match_pairs, src_graph->get_const_nodes(), target_graph->get_const_nodes(), match_instances);
                    fmfusion::IO::save_match_results(target_graph->get_timestamp(),pred_pose, match_instances, src_centroids, ref_centroids, 
                                                    loop_result_dir+"/"+frame_name+".txt");       
                    open3d::io::WritePointCloudToPLY(loop_result_dir+"/"+frame_name+"_src.ply", *cur_active_instance_pcd, {});

                    Visualization::correspondences(src_centroids, ref_centroids, viz.instance_match,LOCAL_AGENT,{},viz.local_frame_offset);                    
                    if(viz.src_map_aligned.getNumSubscribers()>0){
                        O3d_Cloud_Ptr aligned_src_pcd_ptr = std::make_shared<open3d::geometry::PointCloud>(*cur_active_instance_pcd);
                        aligned_src_pcd_ptr->Transform(pred_pose);
                        aligned_src_pcd_ptr->PaintUniformColor({0.0,0.707,0.707});
                        Visualization::render_point_cloud(aligned_src_pcd_ptr, viz.src_map_aligned, LOCAL_AGENT);
                    }
                }

                if(save_corr&& C>0){
                    O3d_Cloud_Ptr corr_src_ptr = std::make_shared<open3d::geometry::PointCloud>(corr_src_points);
                    O3d_Cloud_Ptr corr_ref_ptr = std::make_shared<open3d::geometry::PointCloud>(corr_ref_points);
                    open3d::io::WritePointCloudToPLY(loop_result_dir+"/"+frame_name+"_csrc.ply", *corr_src_ptr, {});
                    open3d::io::WritePointCloudToPLY(loop_result_dir+"/"+frame_name+"_cref.ply", *corr_ref_ptr, {});
                }
                timing_seq.record(tic_toc.toc());// I/O

                loop_count ++;
                prev_loop_frame_id = seq_id;
            }
        }
        
        // Visualization
        if(viz.rgb_image.getNumSubscribers()>0){
            auto rgb_cv = std::make_shared<cv::Mat>(color.height_,color.width_,CV_8UC3);
            memcpy(rgb_cv->data,color.data_.data(),color.data_.size()*sizeof(uint8_t));            
            Visualization::render_image(*rgb_cv, viz.rgb_image, LOCAL_AGENT);
        }
        if(seq_id-broadcast_frame_id>loop_duration){
            Visualization::instance_centroids(semantic_mapping.export_instance_centroids(),viz.src_centroids,LOCAL_AGENT,viz.param.centroid_size);
            Visualization::render_point_cloud(cur_active_instance_pcd, viz.src_graph, LOCAL_AGENT);            
            broadcast_frame_id = seq_id;
        }

        Visualization::render_camera_pose(pose_table[k], viz.camera_pose, LOCAL_AGENT, seq_id);
        Visualization::render_path(pose_table[k], viz.path_msg, viz.path, LOCAL_AGENT, seq_id);

        // update sliding window
        sliding_window.update_translation(seq_id, pose_table[k]);

        ros::spinOnce();
    }
    ROS_WARN("%s Finished sequence", LOCAL_AGENT.c_str());
    ros::shutdown();

    // Post-process
    semantic_mapping.extract_point_cloud();
    semantic_mapping.merge_overlap_instances();
    semantic_mapping.merge_overlap_structural_instances();
    semantic_mapping.extract_bounding_boxes();

    // Save
    semantic_mapping.Save(output_folder+"/"+sequence_name);
    timing_seq.write_log(output_folder+"/"+sequence_name+"/timing.txt");
    comm->write_logs(output_folder+"/"+sequence_name);
    LogWarning("[:s] Save maps and loop results to {:s}", LOCAL_AGENT, output_folder+"/"+sequence_name);

    // ros::spin();
    return 0;
}