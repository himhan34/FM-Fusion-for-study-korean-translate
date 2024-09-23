#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "tf/transform_listener.h"

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>

#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/Eval.h"
#include "tools/g3reg_api.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"
#include "sgloop/LoopDetector.h"
#include "sgloop/Initialization.h"

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

bool isRotationIdentity(const Eigen::Matrix4d &pose)
{
    Eigen::Matrix3d R = pose.block<3,3>(0,0);
    return R.isIdentity(1e-3);
}

/// @brief Take the ref agent info form the communication node. 
///        And update ref graph and implicit features in loop detector accordingly.
bool remote_sg_update(const std::string &target_agent, 
                    const SgCom::AgentDataDict &received_data,
                    std::shared_ptr<fmfusion::LoopDetector> &loop_detector,
                    std::shared_ptr<fmfusion::Graph> &target_graph)
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
            std::cout<<"--- update "<< update_nodes_count<<" ref nodes and " 
                                << update_points_count<<" ref points from "
                                << target_agent<<" ---\n";   
        }

        // std::cout<<"target graph has "<<target_graph->get_const_nodes().size()<<" nodes\n";
        return true;
    }
    else return false;

}

/// @brief Publish the registration transformation to tf tree.
/// @param pose T_parent_child
/// @param parent_frame
/// @param child_frame
void pub_registration_tf(const Eigen::Matrix4d &pose, const std::string &parent_frame, const std::string &child_frame,
                        tf::TransformBroadcaster &br)
{
    // static tf::TransformBroadcaster br_static;

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(pose(0,3),pose(1,3),pose(2,3)));
    tf::Matrix3x3 tf3d;
    tf3d.setValue(pose(0,0),pose(0,1),pose(0,2),
                pose(1,0),pose(1,1),pose(1,2),
                pose(2,0),pose(2,1),pose(2,2));
    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);
    transform.setRotation(tfqt);

    ros::Time time = ros::Time::now() + ros::Duration(0.1);

    br.sendTransform(tf::StampedTransform(transform, time, parent_frame, child_frame));

}

class BrTimer
{
public:
    BrTimer(const std::string &local_agent_name, const std::vector<std::string> &agents,
            ros::NodeHandle &nh)
    {
        local_agent = local_agent_name;

        for(auto agent:agents){
            Eigen::Vector3d init_translation;

            init_translation.x() = nh.param("br/"+agent+"/x", 0.0);
            init_translation.y() = nh.param("br/"+agent+"/y", 0.0);
            init_translation.z() = nh.param("br/"+agent+"/z", 0.0);
            remote_agent_poses[agent] = Eigen::Matrix4d::Identity();
            remote_agent_poses[agent].block<3,1>(0,3) = init_translation;
        }

        pub_alignment = nh.param("br/pub_alignment", false);

    }

    void callback(const ros::TimerEvent &event)
    {
        for(auto agent:remote_agent_poses){
            Eigen::Matrix4d pose = agent.second;
            pub_registration_tf(pose.inverse(), "world", agent.first, br);
        }

    }

    bool set_pred_pose(const std::string &agent, const Eigen::Matrix4d &pose)
    {

        if (pub_alignment &&
            (remote_agent_poses.find(agent)!=remote_agent_poses.end()))
        {
            remote_agent_poses[agent] = pose; // T_remote_local
            return true;
        }
        else return false;
    }

public:
    ros::Timer timer;
    tf::TransformBroadcaster br;


private:
    std::string local_agent;
    std::map<std::string, Eigen::Matrix4d> remote_agent_poses;
    bool pub_alignment;
};

bool tf_listener(const std::string &ref_frame, const std::string &src_frame)
{
    tf::TransformListener listener;
    tf::StampedTransform transform;
    Eigen::Vector3d translation_ref_src;
    std::cout<<"!!! search tf from "<<ref_frame<<" to "<<src_frame<<" !!!\n";
    // std::string ref_frame_name = "/"+ref_frame;
    // std::string src_frame_name = "/"+src_frame;

    try{
        listener.lookupTransform("/"+ref_frame,"/"+src_frame, ros::Time(0), transform);
    }
    catch (tf::TransformException &ex){
        // ROS_ERROR("%s",ex.what());
        return false;
    }
    translation_ref_src.x() = transform.getOrigin().x();
    translation_ref_src.y() = transform.getOrigin().y();
    translation_ref_src.z() = transform.getOrigin().z();

    std::cout<<translation_ref_src.transpose()<<"\n";

    return true;
}

/// @brief Select inliers based on the predicted pose.
/// @return The number of inliers.
int select_inliers(const std::vector<Eigen::Vector3d> &corr_src_points,
                    const std::vector<Eigen::Vector3d> &corr_ref_points,
                    const Eigen::Matrix4d &pose,
                    const float &inlier_radius,
                    std::vector<Eigen::Vector3d> &inlier_src_points,
                    std::vector<Eigen::Vector3d> &inlier_ref_points)
{
    inlier_src_points.clear();
    inlier_ref_points.clear();
    int inlier_count = 0;
    for(int i=0;i<corr_src_points.size();i++){
        Eigen::Vector3d src_point = pose.block<3,3>(0,0)*corr_src_points[i]+pose.block<3,1>(0,3);
        double dist = (src_point-corr_ref_points[i]).norm();
        if(dist<inlier_radius){
            inlier_src_points.push_back(corr_src_points[i]);
            inlier_ref_points.push_back(corr_ref_points[i]);
            inlier_count++;
        }

    }
    return inlier_count;
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
    std::string root_dir;
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

    std::string prediction_folder = nh_private.param("prediction_folder", std::string("prediction_no_augment"));
    std::string output_folder = nh_private.param("output_folder", std::string(""));
    int visualization = nh_private.param("visualization", 0);
    float loop_duration = nh_private.param("loop_duration", 20.0);
    bool prune_instances = nh_private.param("prune_instances", true);
    bool debug_mode = nh_private.param("debug_mode", false);
    bool icp_refine = nh_private.param("icp_refine",false);
    int pose_average_window = nh_private.param("pose_average_size",0);
    bool save_corr = nh_private.param("save_corr",true);
    int o3d_verbose_level = nh_private.param("o3d_verbose_level", 2);
    float sliding_widow_translation = nh_private.param("sliding_widow_translation", 100.0);
    float cool_down_sleep = nh_private.param("cool_down_sleep", -1.0);
    bool enable_tf_br = nh_private.param("enable_tf_br", false);
    bool create_gt_iou = nh_private.param("create_gt_iou", false);
    std::string gt_ref_src_file = nh_private.param("gt_ref_src", std::string("")); // gt T_src_ref
    std::string hidden_feat_dir = nh_private.param("hidden_feat_dir", std::string(""));

    // Initialization
    std::string init_src_scene = nh_private.param("init_src_scene", std::string(""));
    std::string init_ref_scene = nh_private.param("init_ref_scene", std::string(""));
    int init_iter = nh_private.param("init_iter", 0);

    // G3reg params
    double verify_voxel = nh_private.param("g3reg/verify_voxel", 0.5);
    double search_radius = nh_private.param("g3reg/search_radius", 0.5);
    double icp_voxel = nh_private.param("g3reg/icp_voxel", 0.2);
    double ds_voxel = nh_private.param("g3reg/ds_voxel", 0.5);
    int ds_num = nh_private.param("g3reg/ds_num", 9);
    float inlier_ratio_threshold = nh_private.param("g3reg/ir_threshold", 0.3);
    int max_corr_number = nh_private.param("g3reg/max_corr_number", 1000);
    float inlier_radius = nh_private.param("g3reg/inlier_radius", 0.2);
    bool enable_coarse_gnc = nh_private.param("g3reg/enable_coarse_gnc", false);
    float nms_thd = nh_private.param("g3reg/nms_thd", 0.05);

    // Remote agents    
    assert(set_2nd_agent && set_2nd_agent_scene);
    std::vector<std::string> remote_agents = {secondAgent};
    std::vector<std::string> remote_agents_scenes = {secondAgentScene};
    std::vector<int> remote_agents_loop_frames = {0};
    std::vector<std::string> loop_results_dirs;
    std::vector<Eigen::Matrix4d> remote_agents_pred_poses = {Eigen::Matrix4d::Identity()};
    static tf::TransformBroadcaster br;
    std::vector<Eigen::Vector3d> latest_corr_src, latest_corr_ref;
    std::vector<float> latest_corr_scores;

    if(set_3rd_agent){
        remote_agents.push_back(thirdAgent);
        remote_agents_scenes.push_back(thirdAgentScene);
        remote_agents_loop_frames.push_back(0);
        remote_agents_pred_poses.push_back(Eigen::Matrix4d::Identity());
    }
    std::vector<DenseMatch> dense_match(remote_agents.size());

    // outputs
    std::string sequence_name = *filesystem::GetPathComponents(root_dir).rbegin();
    if(!filesystem::DirectoryExists(output_folder)) filesystem::MakeDirectory(output_folder);
    if(hidden_feat_dir.size()>0 && !filesystem::DirectoryExists(hidden_feat_dir)) 
        filesystem::MakeDirectory(hidden_feat_dir);
    
    for(auto scene_name:remote_agents_scenes){
        std::string loop_result_dir = output_folder+"/"+sequence_name+"/"+scene_name;
        if(!filesystem::DirectoryExists(loop_result_dir)) filesystem::MakeDirectoryHierarchy(loop_result_dir);
        loop_results_dirs.push_back(loop_result_dir);
    }

    BrTimer br_timer(LOCAL_AGENT, remote_agents, n);
    if(enable_tf_br){
        br_timer.timer = n.createTimer(ros::Duration(0.01), &BrTimer::callback, &br_timer);
        br_timer.timer.start();
    }

    //
    Eigen::Matrix4d gt_T_ref_src = Eigen::Matrix4d::Identity(); // src->ref
    if(gt_ref_src_file.size()>0){
        fmfusion::IO::read_transformation(gt_ref_src_file, gt_T_ref_src);
        ROS_WARN("Read gt T_src_ref from %s", gt_ref_src_file.c_str());
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
    G3RegAPI::Config config;
    config.set_noise_bounds({0.2, 0.3});
    config.ds_num = ds_num;
    config.plane_resolution = verify_voxel;
    config.verify_mtd = "plane_based";
    config.search_radius = search_radius;
    config.max_corr_num = max_corr_number;
    config.icp_voxel = icp_voxel;
    config.ds_voxel = ds_voxel;

    ROS_WARN("G3Reg: icp voxel %.3f, ds_num %d, ds_voxel %.3f, plane_resolution %.3f, search_radius %.3f", 
                icp_voxel, ds_num, ds_voxel, verify_voxel, search_radius);

    ROS_WARN("G3Reg: max_corr_num %d, inlier_ratio_threshold %.3f", 
                max_corr_number, inlier_ratio_threshold);

    G3RegAPI g3reg(config);

    // Viz
    Visualization::Visualizer viz(n,nh_private,remote_agents);

    // Running sequence
    // fmfusion::TimingSequence timing_seq("#FrameID: Load Mapping; SgCreate SgCoarse Pub&Sub ShapeEncoder PointMatch; Match Reg Viz");
    fmfusion::TimingSequence map_timing("FrameID: Load Mapping");
    fmfusion::TimingSequence broadcast_timing("#FrameID: SgCreate Encode Pub");
    fmfusion::TimingSequence loop_timing("#FrameID: Subscribe ShapeEncoder ShapeModule C-Match D-Match G3Reg ICP I/O");
    
    robot_utils::TicToc tic_toc;
    robot_utils::TicToc tictoc_bk;

    // Reference map
    std::vector<GraphPtr> remote_graphs;
    std::vector<O3d_Cloud_Ptr> remote_pcds;
    std::vector<std::vector<gtsam::Pose3>> remote_pred_poses;
    open3d::utility::Timer timer;
    for(int k=0;k<remote_agents.size();k++){
        remote_graphs.push_back(std::make_shared<Graph>(global_config->graph));
        remote_pred_poses.push_back(std::vector<gtsam::Pose3>());
        remote_pcds.push_back(std::make_shared<open3d::geometry::PointCloud>());
    }

    // Active scene graph
    std::string frame_name;
    open3d::geometry::Image depth, color;
    O3d_Cloud_Ptr cur_active_instance_pcd;
    fmfusion::DataDict src_data_dict;
    int Ns;
    int loop_count = 0;
    bool broadcast_coarse_mode = true; // If been request dense, set to false;
    bool local_coarse_mode = true; // If the local agent detect a loop, set to false;
    bool render_initiated = false;
    sgloop_ros::SlidingWindow sliding_window(sliding_widow_translation);

    ROS_WARN("[%s] Ready to run sequence", LOCAL_AGENT.c_str());
    ros::Duration(1.0).sleep();

    // Init and warm-up the loop detector
    if(init_iter>0)
    {
        ROS_WARN("Activate with %s and %s", init_src_scene.c_str(), init_ref_scene);

        std::shared_ptr<fmfusion::Graph> init_graph_src, init_graph_ref;
        assert(init_src_scene.size()>0 && init_ref_scene.size()>0);
        
        for (int itr =0;itr<init_iter;itr++){
            ROS_WARN("Activation: %d", itr);
            float init_shape_timing;
            init_scene_graph(*global_config, init_src_scene, init_graph_src);
            init_scene_graph(*global_config, init_ref_scene, init_graph_ref);
            loop_detector->encode_src_scene_graph(init_graph_src->get_const_nodes());
            loop_detector->encode_ref_scene_graph(remote_agents[0], init_graph_ref->get_const_nodes());
            loop_detector->encode_concat_sgs(remote_agents[0], init_graph_ref->get_const_nodes().size(),
                                            init_graph_ref->extract_data_dict(),
                                            init_graph_src->get_const_nodes().size(), 
                                            init_graph_src->extract_data_dict(),
                                            init_shape_timing, 
                                            true);
            ROS_WARN("Activate shape encoder takes %.3f ms", init_shape_timing);
        
        }
        ROS_WARN("Activation done");
    }

    for(int k=0;k<rgbd_table.size();k++){
        RGBDFrameDirs frame_dirs = rgbd_table[k];
        frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
        frame_name = frame_name.substr(0,frame_name.find_last_of("."));
        int seq_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));
        if((seq_id-prev_frame_id)<frame_gap) continue;
        map_timing.create_frame(seq_id);

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
        }
        map_timing.record(tic_toc.toc()); 

        if(loaded){
            semantic_mapping.integrate(seq_id,rgbd, pose_table[k], detections);
            prev_frame_id = seq_id;
        }
        map_timing.record(tic_toc.toc()); // mapping
        map_timing.finish_frame();

        if(seq_id-broadcast_frame_id > loop_duration){ // Broadcast iter

            valid_names.clear();
            valid_instances.clear();
            semantic_mapping.export_instances(valid_names, valid_instances, 
                                                sliding_window.get_window_start_frame());
            cur_active_instance_pcd = semantic_mapping.export_global_pcd(true,0.05);
            // std::cout<<"window start frame: "<<sliding_window.get_window_start_frame()<<", "
            //             << valid_names.size()<<" active instances\n";

            // Local encode and broadcast
            tic_toc.tic();
            if(valid_names.size()>global_config->loop_detector.lcd_nodes){ // Local ready to detect loop
                ROS_WARN("** [%s] broadcast frame id: %d, active nodes: %d **", 
                        LOCAL_AGENT.c_str(), seq_id, valid_names.size());
                broadcast_timing.create_frame(seq_id);

                // Explicit local
                Ns = 0;
                src_graph->clear();
                src_graph->initialize(valid_instances);
                src_graph->construct_edges();
                src_graph->construct_triplets();

                if(comm->get_pub_dense_msg()){ 
                    // Been called. Extract dense msg to broadcast. 
                    broadcast_coarse_mode = false;
                    comm->reset_pub_dense_msg();
                    comm->pub_dense_frame_id = seq_id;
                    ROS_WARN("%s been called and set to dense mode", LOCAL_AGENT.c_str());
                }

                src_data_dict.clear();
                bool coarse_mode = broadcast_coarse_mode && local_coarse_mode;
                src_data_dict = src_graph->extract_data_dict(coarse_mode);
                broadcast_timing.record(tic_toc.toc());

                // Implicit local
                loop_detector->encode_src_scene_graph(src_graph->get_const_nodes());
                broadcast_timing.record(tic_toc.toc());

                // Comm
                std::vector<std::vector<float>> src_node_feats_vec;

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
                        broadcast_coarse_mode = true;
                        ROS_WARN("%s Broadcast dense graph", LOCAL_AGENT.c_str());
                        if(remote_graphs[0]->get_timestamp()<0.0) // never received any remote graph
                            ros::Duration(dense_m_config.broadcast_sleep_time).sleep();
                    }
                }
                broadcast_timing.record(tic_toc.toc());// broadcast
                broadcast_timing.finish_frame();

                if(create_gt_iou){ // Save global pcd for post-evaluation
                    open3d::io::WritePointCloudToPLY(output_folder+"/"+sequence_name+"/"+frame_name+".ply", 
                                                    *cur_active_instance_pcd, {});
                }

                if(cool_down_sleep>0.0 && secondAgentScene=="fakeScene") 
                    ros::Duration(cool_down_sleep).sleep();
                
                if(hidden_feat_dir!=""){
                    loop_detector->save_middle_features(hidden_feat_dir+"/"+frame_name+"_sgnet.pt");
                }

            }
        

        }   

        tic_toc.tic();
        if(valid_names.size()>global_config->loop_detector.lcd_nodes){ 
            int target_agent_id = -1;
            int Nr;
            O3d_Cloud_Ptr ref_cloud_ptr(new open3d::geometry::PointCloud()); // used in registration
            loop_timing.create_frame(seq_id);

            for(int z=0;z<remote_agents.size();z++){ //comm update iter

                std::string target_agent = remote_agents[z];
                GraphPtr target_graph = remote_graphs[z];

                const SgCom::AgentDataDict &received_data = comm->get_remote_agent_data(target_agent);
                bool sg_updated = remote_sg_update(target_agent, received_data, loop_detector, target_graph);     

                if(received_data.xyz.size()>0){
                    ref_cloud_ptr->points_.resize(received_data.xyz.size());
                    for(int i=0;i<received_data.xyz.size();i++)
                        ref_cloud_ptr->points_[i] = received_data.xyz[i]; 
                    ref_cloud_ptr = ref_cloud_ptr->VoxelDownSample(0.05);
                    remote_pcds[z]->Clear();
                    // remote_pcds[z]->points_ = ref_cloud_ptr->points_;  
                    remote_pcds[z]->points_.reserve(ref_cloud_ptr->points_.size());
                    for(auto p:ref_cloud_ptr->points_)
                        remote_pcds[z]->points_.push_back(p);
                }

                // Stop at the target agent if condition satifisfied.
                if(sg_updated && (seq_id-remote_agents_loop_frames[z]) > loop_duration){
                    target_agent_id = z;
                    break;
                }
            }

            loop_timing.record(tic_toc.toc());// Sub

            if(target_agent_id >= 0){
                std::string target_agent = remote_agents[target_agent_id];
                GraphPtr target_graph = remote_graphs[target_agent_id];
                std::string loop_result_dir = loop_results_dirs[target_agent_id];
                std::vector<gtsam::Pose3> &poses = remote_pred_poses[target_agent_id];
                DenseMatch &target_dense_m = dense_match[target_agent_id];
                int &prev_loop_frame_id = remote_agents_loop_frames[target_agent_id]; 
                std::cout<<"*** ["<< LOCAL_AGENT<<"] Find " << target_agent
                            <<" Loop iteration "<<seq_id<<" ***\n";
                if(cool_down_sleep>0.0) ros::Duration(cool_down_sleep).sleep();                

                { // Shape encode
                    DataDict ref_data_dict = target_graph->extract_data_dict();
                    Nr = ref_data_dict.instances.size();
                    float shape_encoding_time;
                    std::string shape_feats_dir="";
                    if(hidden_feat_dir!="") shape_feats_dir = hidden_feat_dir+"/"+frame_name+"_shape.pt";
                    bool shape_ret = loop_detector->encode_concat_sgs(target_agent,
                                                    Nr, ref_data_dict, 
                                                    Ns, src_data_dict, 
                                                    shape_encoding_time,
                                                    true,
                                                    shape_feats_dir);
                    loop_timing.record(shape_encoding_time);// shape encode
                    
                    if(shape_ret) { // reset
                        local_coarse_mode = true; // src graph back to coarse mode
                    }
                    
                }
                loop_timing.record(tic_toc.toc());// shape encode

                // Loop closure
                std::vector<NodePair> match_pairs, pruned_match_pairs;
                std::vector<float> match_scores, pruned_match_scores;
                int M=0; // matches
                int Mp=0; // pruned matches
                if(Nr>global_config->loop_detector.lcd_nodes){ // coarse match
                    std::string node_feat_dir = "";
                    if(hidden_feat_dir!="") node_feat_dir = hidden_feat_dir+"/"+frame_name;
                    M = loop_detector->match_nodes(target_agent,
                                                    match_pairs, match_scores, 
                                                    true,
                                                    node_feat_dir);
                }
                std::cout<<"Find "<<M<<" instance matches\n";
                loop_timing.record(tic_toc.toc());// match

                // Points Match
                int C = 0;
                std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
                std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;
                std::vector<int> corr_match_indices;
                std::vector<float> corr_scores_vec;
                fmfusion::O3d_Cloud_Ptr src_cloud_ptr = cur_active_instance_pcd;

                if(M>global_config->loop_detector.recall_nodes){ // Dense mode
                    target_dense_m.coarse_loop_number++;

                    // Prune
                    std::vector<bool> pruned_true_masks(M, false);
                    Registration::pruneInsOutliers(global_config->reg, 
                                            src_graph->get_const_nodes(), target_graph->get_const_nodes(), 
                                            match_pairs, pruned_true_masks);
                    
                    pruned_match_pairs = fmfusion::utility::update_masked_vec(match_pairs, pruned_true_masks);
                    pruned_match_scores = fmfusion::utility::update_masked_vec(match_scores, pruned_true_masks);
                    Mp = pruned_match_pairs.size();
                    std::cout<<"Keep "<<Mp<<" consistent matched nodes\n";

                    // Dense Match                    
                    if(loop_detector->IsSrcShapeEmbedded() && loop_detector->IsRefShapeEmbedded(target_agent)){
                        std::string point_feat_dir = "";
                        if(hidden_feat_dir!="") point_feat_dir = hidden_feat_dir+"/"+frame_name;
                        C = loop_detector->match_instance_points(target_agent,
                                                                pruned_match_pairs, 
                                                                corr_src_points, 
                                                                corr_ref_points, 
                                                                corr_match_indices,
                                                                corr_scores_vec,
                                                                point_feat_dir);
                    }
                }
                else{
                    pruned_match_pairs = match_pairs;
                    pruned_match_scores = match_scores;
                    Mp = M;
                }
                loop_timing.record(tic_toc.toc());// Point match
                fmfusion::IO::extract_instance_correspondences(src_graph->get_const_nodes(), 
                                                            target_graph->get_const_nodes(), 
                                                            pruned_match_pairs, pruned_match_scores, 
                                                            src_centroids, ref_centroids);  

                // Pose Estimation
                Eigen::Matrix4d pred_pose;
                pred_pose.setIdentity();
                tic_toc.tic();
                if(M>global_config->loop_detector.recall_nodes){
                    tictoc_bk.tic();
                    
                    double inlier_ratio = 0.0;
                    if(corr_src_points.size()>0){ // Dense registration
                        timer.Start();
                        downsample_corr_nms(corr_src_points, corr_ref_points, corr_scores_vec, nms_thd);
                        downsample_corr_topk(corr_src_points, corr_ref_points, corr_scores_vec, ds_voxel, max_corr_number);

                        inlier_ratio = g3reg.estimate_pose_gnc(src_centroids,ref_centroids,
                                            corr_src_points, corr_ref_points,
                                            corr_scores_vec);
                                            
                        if(inlier_ratio<inlier_ratio_threshold){
                            g3reg.estimate_pose(src_centroids, 
                                                ref_centroids,
                                                corr_src_points,
                                                corr_ref_points,
                                                corr_scores_vec,
                                                src_cloud_ptr,
                                                ref_cloud_ptr);
                            ROS_WARN("Dnese Reg by G3Reg\n");
                        }
                        timer.Stop();
                        float reg_duration = timer.GetDurationInMillisecond();
                        ROS_WARN("Dense reg inlier ratio: %f, corr_src: %d, corr_ref: %d, srcpcd: %d, refpcd: %d, time: %.3f", 
                                inlier_ratio,
                                corr_src_points.size(),
                                corr_ref_points.size(),
                                src_cloud_ptr->points_.size(),
                                ref_cloud_ptr->points_.size(),
                                reg_duration);
                    }
                    else{ // Coarse registration
                        if(enable_coarse_gnc){
                            inlier_ratio = g3reg.estimate_pose_gnc(src_centroids, ref_centroids,
                                                                    latest_corr_src, 
                                                                    latest_corr_ref,
                                                                    latest_corr_scores);
                            if(inlier_ratio<inlier_ratio_threshold){
                                g3reg.estimate_pose(src_centroids, ref_centroids,
                                                latest_corr_src, latest_corr_ref,
                                                latest_corr_scores,
                                                src_cloud_ptr, 
                                                remote_pcds[target_agent_id]);
                            }
                        }
                        else
                            g3reg.estimate_pose(src_centroids, ref_centroids,
                                                latest_corr_src, latest_corr_ref,
                                                latest_corr_scores,
                                                src_cloud_ptr, 
                                                remote_pcds[target_agent_id]);

                        ROS_WARN("Total corr: %d, Ref points have %d points\n", 
                                corr_src_points.size(),
                                remote_pcds[target_agent_id]->points_.size());
                    }

                    pred_pose = g3reg.reg_result.tf;
                    if((pred_pose - Eigen::Matrix4d::Identity()).norm()<1e-3){
                        ROS_WARN("G3Reg calles a failed registration.");
                        // Cancel the called loop
                        M  = 0;
                        Mp = 0;
                        target_dense_m.coarse_loop_number --;
                    }
                                    
                    if(corr_src_points.size()>0 && M>0){ // Save the latest dense correspondences
                        int count_inliers = select_inliers(corr_src_points, corr_ref_points, pred_pose, inlier_radius,
                                        latest_corr_src, latest_corr_ref);
                        latest_corr_scores.clear();
                        latest_corr_scores.resize(count_inliers,0.2);
                        ROS_WARN("Select %d inliers\n", count_inliers);
                    }
                    
                    loop_timing.record(tic_toc.toc());// G3Reg

                    if(icp_refine && ref_cloud_ptr->HasPoints()){
                        if(!ref_cloud_ptr->HasNormals()) ref_cloud_ptr->EstimateNormals();
                        pred_pose = g3reg.icp_refine(src_cloud_ptr, ref_cloud_ptr, pred_pose);
                        ROS_WARN("ICP refine pose");
                    } 
                    loop_timing.record(tic_toc.toc());// ICP

                    if(pose_average_window>1){ // abandon
                        assert(false);
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

                    br_timer.set_pred_pose(target_agent, pred_pose);
                    remote_agents_pred_poses[target_agent_id] = pred_pose;
                }
                else{
                    loop_timing.record(tic_toc.toc());// fake G3Reg
                    loop_timing.record(0.00001);// fake ICP
                }

                if(target_dense_m.coarse_loop_number>dense_m_config.min_loops &&
                    (seq_id - target_dense_m.last_frame_id)>dense_m_config.min_frame_gap){ // Send ONE request message.
                    comm->send_request_dense(target_agent);
                    local_coarse_mode = false;
                    target_dense_m.coarse_loop_number = 0;
                    target_dense_m.last_frame_id = seq_id;
                    ROS_WARN("%s Request dense message at %s",LOCAL_AGENT.c_str(), frame_name.c_str());
                }

                // I/O
                std::vector <std::pair<fmfusion::InstanceId, fmfusion::InstanceId>> match_instances;
                if(M>global_config->loop_detector.recall_nodes){ //IO              
                    Eigen::Matrix4d T_local_remote;
                    if(viz.Transfrom_local_remote.find(target_agent)==viz.Transfrom_local_remote.end())
                        T_local_remote = Eigen::Matrix4d::Identity();
                    else T_local_remote = viz.Transfrom_local_remote[target_agent];

                    std::vector<bool> pred_masks;
                    float DIST_THRESHOLD;
                    if(C>0){ // Viz dense matches
                        if(C>500){
                            std::vector<int> sample_indices;
                            std::cout<<"todo\n";
                        }
                        
                        if(gt_ref_src_file.size()>0){
                            DIST_THRESHOLD = 0.5;
                            fmfusion::mark_tp_instances(gt_T_ref_src, 
                                                        corr_src_points, corr_ref_points,
                                                        pred_masks, DIST_THRESHOLD);
                        }
                        else pred_masks.resize(corr_src_points.size(), true);


                        Visualization::correspondences(corr_src_points, corr_ref_points, 
                                                    viz.instance_match,LOCAL_AGENT,pred_masks,
                                                    T_local_remote);
                        
                    }
                    else{ // Viz coarse matches
                        if(gt_ref_src_file.size()>0){
                            DIST_THRESHOLD = 1.0;
                            fmfusion::mark_tp_instances(gt_T_ref_src, 
                                                        src_centroids, ref_centroids, 
                                                        pred_masks, DIST_THRESHOLD);
                        }
                        else pred_masks.resize(src_centroids.size(), true);
                        
                        Visualization::correspondences(src_centroids, ref_centroids, 
                                                    viz.instance_match,LOCAL_AGENT,pred_masks,
                                                    T_local_remote);                        
                    }

                    // todo: save at valid frames
                    if(latest_corr_ref.size()>0){

                        fmfusion::IO::extract_match_instances(pruned_match_pairs, 
                                                            src_graph->get_const_nodes(), 
                                                            target_graph->get_const_nodes(), 
                                                            match_instances);
                        fmfusion::IO::save_match_results(target_graph->get_timestamp(),pred_pose, match_instances, src_centroids, ref_centroids, 
                                                        loop_result_dir+"/"+frame_name+".txt");       
                        fmfusion::IO::save_graph_centroids(src_graph->get_centroids(), 
                                                        target_graph->get_centroids(), 
                                                        loop_result_dir+"/"+frame_name+"_centroids.txt");
                        
                        open3d::io::WritePointCloudToPLY(loop_result_dir+"/"+frame_name+"_src.ply", *cur_active_instance_pcd, {});
                                                    
                        if(viz.src_map_aligned.getNumSubscribers()>0){
                            O3d_Cloud_Ptr aligned_src_pcd_ptr = std::make_shared<open3d::geometry::PointCloud>(*cur_active_instance_pcd);
                            aligned_src_pcd_ptr->Transform(pred_pose);
                            aligned_src_pcd_ptr->PaintUniformColor({0.0,0.707,0.707});
                            Visualization::render_point_cloud(aligned_src_pcd_ptr, viz.src_map_aligned, target_agent);
                        }
                    
                        if(save_corr){
                            O3d_Cloud_Ptr corr_src_ptr, corr_ref_ptr;
                            if(C>0){       
                                corr_src_ptr = std::make_shared<open3d::geometry::PointCloud>(corr_src_points);
                                corr_ref_ptr = std::make_shared<open3d::geometry::PointCloud>(corr_ref_points);
                                fmfusion::IO::save_corrs_match_indices(corr_match_indices, 
                                                                    corr_scores_vec,
                                                                    loop_result_dir+"/"+frame_name+"_cmatches.txt");                        
                            }
                            else{
                                corr_src_ptr = std::make_shared<open3d::geometry::PointCloud>(latest_corr_src);
                                corr_ref_ptr = std::make_shared<open3d::geometry::PointCloud>(latest_corr_ref);
                                ROS_WARN("Save latest correspondences %ld\n", latest_corr_src.size());
                            }

                            open3d::io::WritePointCloudToPLY(loop_result_dir+"/"+frame_name+"_csrc.ply", *corr_src_ptr, {});
                            open3d::io::WritePointCloudToPLY(loop_result_dir+"/"+frame_name+"_cref.ply", *corr_ref_ptr, {});
                        }
                    }

                }
                else{
                    fmfusion::IO::save_match_results(target_graph->get_timestamp(),
                                                    Eigen::Matrix4d::Identity(), 
                                                    match_instances, 
                                                    {}, {}, 
                                                    loop_result_dir+"/"+frame_name+".txt");
                }
                loop_timing.record(tic_toc.toc());// I/O
                loop_timing.finish_frame();

                loop_count ++;
                prev_loop_frame_id = seq_id;
            }
        }
        
        if(seq_id-broadcast_frame_id>loop_duration && secondAgentScene=="fakeScene"){ // Save src pcd
            open3d::io::WritePointCloudToPLY(output_folder+"/"+sequence_name+"/fakeScene/"+frame_name+"_src.ply", 
                                            *cur_active_instance_pcd, {});  
        }    

        // Visualization
        if(viz.rgb_image.getNumSubscribers()>0){
            auto rgb_cv = std::make_shared<cv::Mat>(color.height_,color.width_,CV_8UC3);
            memcpy(rgb_cv->data,color.data_.data(),color.data_.size()*sizeof(uint8_t));            
            Visualization::render_image(*rgb_cv, viz.rgb_image, LOCAL_AGENT);
        }
        if(seq_id-broadcast_frame_id>loop_duration){
            std::vector<Eigen::Vector3d> src_instance_centroids;
            std::vector<std::string> src_instance_annotations;

            src_instance_centroids = semantic_mapping.export_instance_centroids(0);
            src_instance_annotations = semantic_mapping.export_instance_annotations(0);
            // std::cout<<"Extract "<<src_instance_centroids.size()<<" centroids\n";

            Visualization::instance_centroids(src_instance_centroids,
                                            viz.src_centroids,
                                            LOCAL_AGENT,
                                            viz.param.centroid_size,
                                            viz.param.centroid_color);
            
            Visualization::node_annotation(src_instance_centroids,
                                            src_instance_annotations,
                                            viz.node_annotation,
                                            LOCAL_AGENT,
                                            viz.param.annotation_size,
                                            viz.param.annotation_voffset,
                                            viz.param.annotation_color);

            Visualization::render_point_cloud(cur_active_instance_pcd, viz.src_graph, LOCAL_AGENT);            
            broadcast_frame_id = seq_id;
            if(!render_initiated) ros::Duration(2.0).sleep();

            render_initiated = true;
        }
        if(viz.pred_image.getNumSubscribers()>0){
            std::string pred_img_dir = root_dir+"/pred_viz/"+frame_name+".jpg";
            cv::Mat pred_img = cv::imread(pred_img_dir);
            Visualization::render_image(pred_img, viz.pred_image, LOCAL_AGENT);
        }

        Visualization::render_camera_pose(pose_table[k], viz.camera_pose, LOCAL_AGENT, seq_id);
        Visualization::render_path(pose_table[k], viz.path_msg, viz.path, LOCAL_AGENT, seq_id);

        // update sliding window
        sliding_window.update_translation(seq_id, pose_table[k]);

        ros::spinOnce();
    }

    //
    if(viz.path_aligned.getNumSubscribers()>0){
        nav_msgs::Path aligned_path_msg;

        Visualization::render_path(pose_table, 
                                    remote_agents_pred_poses[0], 
                                    remote_agents[0], 
                                    aligned_path_msg, 
                                    viz.path_aligned);
        Eigen::Matrix4d aligned_last_pose = remote_agents_pred_poses[0] * pose_table.back();
        Visualization::render_camera_pose(aligned_last_pose, viz.camera_pose, remote_agents[0], 0);
    }

    ROS_WARN("%s Finished sequence at frame %s", LOCAL_AGENT.c_str(), frame_name.c_str());
    ros::shutdown();
    if(create_gt_iou) return 0;

    // Post-process
    semantic_mapping.extract_point_cloud();
    semantic_mapping.merge_floor();
    semantic_mapping.merge_overlap_instances();
    // semantic_mapping.merge_overlap_structural_instances();
    semantic_mapping.extract_bounding_boxes();

    // Save
    semantic_mapping.Save(output_folder+"/"+sequence_name);
    map_timing.write_log(output_folder+"/"+sequence_name+"/map_timing.txt");
    broadcast_timing.write_log(output_folder+"/"+sequence_name+"/broadcast_timing.txt");
    loop_timing.write_log(output_folder+"/"+sequence_name+"/loop_timing.txt");
    // timing_seq.write_log(output_folder+"/"+sequence_name+"/timing.txt");
    comm->write_logs(output_folder+"/"+sequence_name);
    LogWarning("[:s] Save maps and loop results to {:s}", LOCAL_AGENT, output_folder+"/"+sequence_name);

    // ros::spin();
    return 0;
}