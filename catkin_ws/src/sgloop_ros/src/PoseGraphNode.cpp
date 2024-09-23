#include <sstream>
#include <unordered_map>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "open3d_conversions/open3d_conversions.h"

#include "tools/IO.h"

#include "Visualization.h"
#include "utility/CameraPoseVisualization.h"
#include "mapping/SemanticMapping.h"
#include "sgloop/Graph.h"

typedef std::pair<std::string, std::string> LoopPair;

bool read_entire_camera_poses(const std::string &scene_folder,
                        std::vector<Eigen::Matrix4d> &src_poses_map)
{
    std::vector<fmfusion::IO::RGBDFrameDirs> src_rgbds;
    std::vector<Eigen::Matrix4d> src_poses;

    bool read_ret = fmfusion::IO::construct_preset_frame_table(
        scene_folder,"data_association.txt","trajectory.log",src_rgbds,src_poses);
    if(!read_ret) {
        return false;
    }

    for (int i=0;i<src_rgbds.size();i++){
        src_poses_map.push_back(src_poses[i]);
    }

    return true;
}

bool read_entire_camera_poses(const std::string &scene_folder,
                        std::unordered_map<std::string, Eigen::Matrix4d> &src_poses_map)
{
    std::vector<fmfusion::IO::RGBDFrameDirs> src_rgbds;
    std::vector<Eigen::Matrix4d> src_poses;

    bool read_ret = fmfusion::IO::construct_preset_frame_table(
        scene_folder,"data_association.txt","trajectory.log",src_rgbds,src_poses);
    if(!read_ret) {
        return false;
    }

    for (int i=0;i<src_rgbds.size();i++){
        std::string frame_file_name = open3d::utility::filesystem::GetFileNameWithoutDirectory(src_rgbds[i].first);
        std::string frame_name = frame_file_name.substr(0, frame_file_name.size()-4);
        // std::cout<<frame_name<<std::endl;
        // src_poses_map[src_rgbds[i].first] = src_poses[i];
        src_poses_map[frame_name] = src_poses[i];
    }

    return true;
}

bool read_frames_poses(const std::string &frame_pose_file,
                    std::unordered_map<std::string, Eigen::Matrix4d> &frame_poses)
{
    std::ifstream pose_file(frame_pose_file);
    if (!pose_file.is_open()){
        std::cerr<<"Cannot open pose file: "<<frame_pose_file<<std::endl;
        return false;
    }

    std::string line;
    while (std::getline(pose_file, line)){
        if(line.find("#")!=std::string::npos){
            continue;
        }
        std::istringstream iss(line);
        std::string frame_name;
        Eigen::Vector3d p;
        Eigen::Quaterniond q;
        Eigen::Matrix4d pose;
        iss>>frame_name;
        iss>>p[0]>>p[1]>>p[2]>>q.x()>>q.y()>>q.z()>>q.w();
        pose.setIdentity();
        pose.block<3,1>(0,3) = p;
        pose.block<3,3>(0,0) = q.toRotationMatrix();

        frame_poses[frame_name] = pose;
    }

    return true;
}

bool read_loop_transformations(const std::string &loop_file_dir,
                        std::vector<LoopPair> &loop_pairs,
                        std::vector<Eigen::Matrix4d> &loop_transformations)
{
    std::ifstream loop_file(loop_file_dir);
    if (!loop_file.is_open()){
        std::cerr<<"Cannot open loop file: "<<loop_file_dir<<std::endl;
        return false;
    }

    std::string line;
    while (std::getline(loop_file, line)){
        if(line.find("#")!=std::string::npos){
            continue;
        }
        std::istringstream iss(line);
        Eigen::Vector3d t_vec;
        Eigen::Quaterniond quat;
        std::string src_frame, ref_frame;

        iss>>src_frame>>ref_frame;
        iss>>t_vec[0]>>t_vec[1]>>t_vec[2]>>quat.x()>>quat.y()>>quat.z()>>quat.w();

        Eigen::Matrix4d transformation;
        transformation.setIdentity();

        transformation.block<3,1>(0,3) = t_vec;
        transformation.block<3,3>(0,0) = quat.toRotationMatrix();
        // std::cout<<src_frame<<"->"<<ref_frame<<std::endl;

        loop_pairs.push_back(std::make_pair(src_frame, ref_frame));
        loop_transformations.push_back(transformation);
    }

    std::cout<<"Load "<<loop_transformations.size()<<" loop transformations"<<std::endl;

    return true;
}


bool read_pose_file(const std::string &pose_file_dir,
                    Eigen::Matrix4d &pose)
{
    std::ifstream pose_file(pose_file_dir);
    if (!pose_file.is_open()){
        std::cerr<<"Cannot open pose file: "<<pose_file_dir<<std::endl;
        return false;
    }

    std::string line;
    pose.setIdentity();    
    int row=0;

    while (std::getline(pose_file, line)){
        if(line.find("#")!=std::string::npos){
            continue;
        }
        std::istringstream iss(line);
        iss>>pose(row,0)>>pose(row,1)>>pose(row,2)>>pose(row,3);
        row++;
    }
    return true;
}

int read_our_registration(const std::string &result_folder,
                            std::vector<std::string> &src_frames,
                            std::vector<std::string> &ref_frames,
                            std::vector<Eigen::Matrix4d> &T_ref_srcs)
{
    int MAX_FRAME_NUMBER = 2000;
    std::cout<<result_folder<<std::endl;

    for (int i=0;i<MAX_FRAME_NUMBER;i++){
        std::stringstream frame_dir, frame_name;
        frame_dir<<result_folder<<"/frame-"<<std::setw(6)<<std::setfill('0')<<i<<".txt";
        frame_name<<"frame-"<<std::setw(6)<<std::setfill('0')<<i;

        if(open3d::utility::filesystem::FileExists(frame_dir.str())){
            float ref_timestamp;
            Eigen::Matrix4d T_ref_src;
            std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
            std::vector<Eigen::Vector3d> src_centroids;
            std::vector<Eigen::Vector3d> ref_centroids;

            fmfusion::IO::load_match_results(frame_dir.str(), 
                                            ref_timestamp, 
                                            T_ref_src, 
                                            match_pairs, 
                                            src_centroids, ref_centroids);

            // std::cout<<"Load frame "<<frame_name.str()
            //     <<", ref timestamp: "<<ref_timestamp<<std::endl;
            int ref_frame_id = (ref_timestamp - 12000) * 10; 

            //
            std::stringstream ref_frame_name;
            ref_frame_name<<"frame-"<<std::setw(6)<<std::setfill('0')<<ref_frame_id;
            src_frames.push_back(frame_name.str());
            ref_frames.push_back(ref_frame_name.str());
            T_ref_srcs.push_back(T_ref_src);

            // break;
        }
        // std::cout<<frame_dir.str()<<std::endl;
        
    }

    ROS_WARN("Load %d frames registration result", src_frames.size());

    return src_frames.size();

}

std::vector<bool> sampling_mask(int N, float ratio)
{
    std::vector<bool> mask(N, false);
    int num = N * ratio;
    for (int i=0;i<num;i++){
        mask[i] = true;
    }
    std::random_shuffle(mask.begin(), mask.end());
    return mask;
}


int main(int argc, char **argv)
{
    using namespace open3d::utility;
    using namespace fmfusion;

    ros::init(argc, argv, "LoopNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    ros::Publisher src_camera_pose_pub = nh_private.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
    ros::Publisher tp_camera_pose_pub = nh_private.advertise<visualization_msgs::MarkerArray>("tp_pose_graph", 1000);
    ros::Publisher fp_camera_pose_pub = nh_private.advertise<visualization_msgs::MarkerArray>("fp_pose_graph", 1000);
    ros::Publisher loop_edge_pub = nh_private.advertise<visualization_msgs::Marker>("loop_edge", 1000);
    std::string src_name, ref_name;
    std::string scene_name;
    std::string src_scene_dir, ref_scene_dir;
    std::string scene_result_folder;
    std::string loop_file;
    float camera_color[4];

    // Params
    nh_private.getParam("src_name", src_name);
    nh_private.getParam("ref_name", ref_name);
    nh_private.getParam("scene_name", scene_name);
    nh_private.getParam("src_scene_dir", src_scene_dir);
    nh_private.getParam("ref_scene_dir", ref_scene_dir);
    nh_private.getParam("scene_result_folder", scene_result_folder);
    camera_color[0] = nh_private.param("camera_marker/r", 1.0);
    camera_color[1] = nh_private.param("camera_marker/g", 0.0);
    camera_color[2] = nh_private.param("camera_marker/b", 0.0);
    camera_color[3] = nh_private.param("camera_marker/a", 1.0);

    float camera_scale = nh_private.param("camera_marker/scale", 0.1);
    float camera_line_width = nh_private.param("camera_marker/line_width", 0.01);
    float ate_threshold = nh_private.param("ate_threshold", 0.5);
    bool map_only = nh_private.param("map_only", false);
    int iter_num = nh_private.param("iter_num", 2);
    bool dense_corr = nh_private.param("dense_corr", false);
    std::string map_name = nh_private.param("map_name", std::string(""));
    std::string pose_graph_folder = nh_private.param("pose_graph_folder", std::string(""));
    std::string gt_file = nh_private.param("gt_file", std::string(""));
    std::string frame_file_name = nh_private.param("frame_file_name", std::string("src_poses.txt"));
    std::string select_frame_name = nh_private.param("select_frame", std::string(""));
    std::string loop_filename = nh_private.param("loop_filename", std::string(""));
    std::string hydra_centroid_file = nh_private.param("hydra_centroid_file", std::string(""));
    float corr_sampling_ratio = nh_private.param("corr_sampling_ratio", 0.1);
    int mode = nh_private.param("mode",0);
    bool render_trajectory = nh_private.param("render_trajectory", false);
    bool render_loops = nh_private.param("render_loops", true);
    bool render_pose = nh_private.param("render_pose", true);
    bool render_superpoint = nh_private.param("render_superpoint", false);
    bool pseudo_corr = nh_private.param("pseudo_corr", false);
    bool paint_all_floor = nh_private.param("paint_all_floor", false);
    // 
    Eigen::Vector3d t_src_ref;
    Eigen::Matrix4d T_src_ref_viz = Eigen::Matrix4d::Identity(); // src->ref incorporated for vizualization

    { // Load src->ref transformation
        t_src_ref[0] = nh.param("br/"+ref_name+"/x", 0.0);
        t_src_ref[1] = nh.param("br/"+ref_name+"/y", 0.0);
        t_src_ref[2] = nh.param("br/"+ref_name+"/z", 0.0);
        float yaw_src_ref = nh.param("br/"+ref_name+"/yaw", 0.0);
        T_src_ref_viz.block<3,1>(0,3) = t_src_ref;
        T_src_ref_viz.block<3,3>(0,0) = Eigen::AngleAxisd(yaw_src_ref, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }
    
    // Init viz tool
    Visualization::Visualizer viz(nh, nh_private, {ref_name});
    CameraPoseVisualization camera_pose_viz_Src(camera_color[0], camera_color[1], camera_color[2], camera_color[3]);
    camera_pose_viz_Src.setScale(camera_scale);
    camera_pose_viz_Src.setLineWidth(camera_line_width);
    std::vector<Eigen::Vector3d> src_centroids_m, ref_centroids_m;
    std::vector<Eigen::Vector3d> src_corrs_points, ref_corrs_points;
    std::vector<bool> corr_mask;
    std::vector<std::pair<uint32_t,uint32_t>> node_pairs;
    std::vector<bool> node_tp_masks;
    std::vector<bool> loop_correct_masks;
    std::vector<Eigen::Vector3d> loop_src_point, loop_ref_point;

    // Load map
    Eigen::Matrix4d T_gt_ref_src;   
    Eigen::Matrix4d T_ref_src_pred = Eigen::Matrix4d::Identity();
     
    std::vector<Eigen::Vector3d> src_centroids;
    std::vector<std::string> src_instance_annotations;
    std::shared_ptr<open3d::geometry::PointCloud> global_src_pcd;
    // Visualization
    std_msgs::Header header_msg;
    header_msg.frame_id = src_name;
    header_msg.stamp = ros::Time::now();

    // Visualize registered camera poses
    CameraPoseVisualization camera_pose_pnp_true(0.0, 1.0, 0.0, 1.0);
    CameraPoseVisualization camera_pose_pnp_false(1.0, 0.0, 0.0, 1.0);
    camera_pose_pnp_true.setScale(camera_scale);
    camera_pose_pnp_true.setLineWidth(camera_line_width);
    camera_pose_pnp_false.setScale(camera_scale);
    camera_pose_pnp_false.setLineWidth(camera_line_width);

    // Load scene graph
    std::shared_ptr<Graph> src_graph;
    std::unordered_map<std::string, Eigen::Matrix4d> src_poses; // {frame_name: pose}

    {
        std::vector<InstanceId> src_names;
        std::vector<InstancePtr> src_instances;   
        std::string config_file;
        if(render_pose){
            read_entire_camera_poses(src_scene_dir, src_poses);
            std::cout<<"Load "<<src_poses.size()<<" poses"<<std::endl;  
        }
        bool set_cfg = nh_private.getParam("cfg_file", config_file);
        assert(set_cfg);
        Config *sg_config = utility::create_scene_graph_config(config_file, false);
        if(gt_file.size()>0){
            read_pose_file(gt_file, T_gt_ref_src); // gt src->ref
        }
        else T_gt_ref_src.setIdentity();
        std::cout<<"Load scene relative pose from "<<gt_file<<std::endl;
        Eigen::Quaterniond q_gt(T_gt_ref_src.block<3,3>(0,0));
        std::cout<<std::fixed<<std::setprecision(6)
                <<T_gt_ref_src(0,3)<<" "<<T_gt_ref_src(1,3)<<" "<<T_gt_ref_src(2,3)<<" "
                <<q_gt.x()<<" "<<q_gt.y()<<" "<<q_gt.z()<<" "<<q_gt.w()<<" "
                <<std::endl;

        auto src_mapping = std::make_shared<SemanticMapping>(SemanticMapping(sg_config->mapping_cfg,sg_config->instance_cfg));
        src_mapping->load(src_scene_dir);
        src_mapping->extract_bounding_boxes();
        src_mapping->export_instances(src_names,src_instances);
        global_src_pcd = src_mapping->export_global_pcd(true,0.05);

        src_graph = std::make_shared<Graph>(sg_config->graph);
        src_graph->initialize(src_instances);
        src_graph->construct_edges();
        src_graph->construct_triplets();

        src_centroids = src_graph->get_centroids();
        auto src_nodes = src_graph->get_const_nodes();

        for(auto node: src_nodes)
            src_instance_annotations.push_back(node->semantic);
        
        if(hydra_centroid_file.size()>0){
            assert(render_superpoint==false);
            src_centroids.clear();
            src_instance_annotations.clear();
            fmfusion::IO::load_instance_info(hydra_centroid_file, 
                                            src_centroids, src_instance_annotations);
            ROS_WARN("Load %d centroids from %s", src_centroids.size(), hydra_centroid_file.c_str());
        }
        for(auto &center:src_centroids)
            center = T_gt_ref_src.block<3,3>(0,0) * center + T_gt_ref_src.block<3,1>(0,3);

        // assert(src_centroids.size()==src_instance_annotations.size());
        // std::cout<<"Load "<<src_centroids.size()<<" centroids and "
        //         <<src_instance_annotations.size()<<" annotations"<<std::endl;
        
        if(render_superpoint){
            src_instance_annotations.clear();
            src_centroids.clear();

            auto src_sp = open3d::io::CreatePointCloudFromFile(scene_result_folder+"/sp_"+scene_name+".ply");
            src_sp->Transform(T_gt_ref_src);
            src_centroids = src_sp->points_;
        }

    }

    if (map_name!=""){
        global_src_pcd = open3d::io::CreatePointCloudFromFile(src_scene_dir+"/"+map_name);
        ROS_WARN("%s loads global map from %s and overwrite", 
                    src_name.c_str(),(src_scene_dir+"/"+map_name).c_str());
    }

    if(mode==1 || mode==15 || mode==99){ // Render src camera poses
        const int frame_gap_ = 10;
        int idx = 0;
        for(const auto &pose:src_poses){ // Src camera poses
            if(idx%frame_gap_!=0){
                idx++;
                continue;
            }
            Eigen::Matrix4d viz_pose = T_gt_ref_src * pose.second;

            Eigen::Vector3d p = viz_pose.block<3,1>(0,3);
            Eigen::Quaterniond q(viz_pose.block<3,3>(0,0));
            camera_pose_viz_Src.add_pose(p, q);
            idx++;
        }
    }

    // Load
    if(mode==0){ // matches only
        if (paint_all_floor){
            Eigen::Vector3d floor_color(200,200,200);
            src_graph->paint_all_floor(floor_color/255.0);
        }

        if(map_name==""){
            // global_src_pcd->Clear();
            global_src_pcd = src_graph->extract_global_cloud();
            global_src_pcd->Transform(T_gt_ref_src);
        }



        if(scene_result_folder.size()>0){ // Load node match results
            std::cout<<"Load match results from "<<scene_result_folder<<std::endl;
            std::string file_name = "/node_matches.txt";
            if (render_superpoint)
                file_name = "/superpoint_matches.txt";

            fmfusion::IO::load_node_matches(scene_result_folder+file_name,
                                            node_pairs, node_tp_masks,
                                            src_centroids_m, ref_centroids_m);
            if(dense_corr){
                std::string src_corr_filname, ref_corr_filename;
                std::string corr_mask_filename;
                if (pseudo_corr){
                    src_corr_filname = scene_result_folder+"/pseudo_corr_src.ply";
                    ref_corr_filename = scene_result_folder+"/pseudo_corr_ref.ply";
                    corr_mask_filename = scene_result_folder+"/pseudo_corres_pos.txt";
                }
                else{
                    src_corr_filname = scene_result_folder+"/corr_src.ply";
                    ref_corr_filename = scene_result_folder+"/corr_ref.ply";
                    corr_mask_filename = scene_result_folder+"/corres_pos.txt";
                }


                auto corr_src_pcd = open3d::io::CreatePointCloudFromFile(src_corr_filname);
                auto corr_ref_pcd = open3d::io::CreatePointCloudFromFile(ref_corr_filename);
                corr_src_pcd->Transform(T_gt_ref_src);
                auto src_corrs_points_ = corr_src_pcd->points_;
                auto ref_corrs_points_ = corr_ref_pcd->points_;
                std::vector<bool> corr_mask_;
                std::cout<<"Load "<<src_corrs_points_.size()<<" dense correspondences"<<std::endl;

                if(!fmfusion::IO::load_single_col_mask(corr_mask_filename,corr_mask_)){
                    corr_mask_.resize(src_corrs_points_.size(), true);
                    std::cout<<"Set all point correspondences as correct"<<std::endl;
                }

                int C = src_corrs_points_.size();
                std::vector<bool> corr_samples = sampling_mask(C, corr_sampling_ratio);
                for (int k=0;k<C;k++){ // sampling mask
                    if(corr_samples[k]){
                        src_corrs_points.push_back(src_corrs_points_[k]);
                        ref_corrs_points.push_back(ref_corrs_points_[k]);
                        corr_mask.push_back(corr_mask_[k]);
                    }


                }

            }

        }

    }
    else if (mode==1){ // Our aligned pose graph 
        ROS_WARN("[MODE%d] Render predicted relative camera poses from %s", mode, pose_graph_folder.c_str());
        std::vector<std::string> src_frames, ref_frames;
        std::vector<Eigen::Matrix4d> T_ref_srcs;
        std::unordered_map<std::string, Eigen::Matrix4d> ref_poses;
        std::vector<Eigen::Vector3d> src_positions, ref_positions;
        std::vector<bool> loop_tp_masks;

        read_our_registration(pose_graph_folder, src_frames, ref_frames, T_ref_srcs);
        read_entire_camera_poses(ref_scene_dir, ref_poses);
        std::cout<<"Load "<<ref_poses.size()<<" ref camera poses"<<std::endl;

        int frame_gap = 10;
        int idx = 0;

        for(const auto &pose:src_poses){ // Src camera poses
            if(idx%frame_gap!=0){
                idx++;
                continue;
            }
            Eigen::Matrix4d viz_pose = pose.second;

            Eigen::Vector3d p = viz_pose.block<3,1>(0,3);
            Eigen::Quaterniond q(viz_pose.block<3,3>(0,0));
            camera_pose_viz_Src.add_pose(p, q);
            idx++;
        }

        int M = ref_frames.size();
        int count_true = 0;
        int count_fp = 0;
        std::stringstream msg;
        msg<<"false frames: ";
        for(int m=0;m<M;m++){ // Align camera pose into src frame using registration result
            if(ref_poses.find(ref_frames[m])==ref_poses.end()){
                ROS_WARN("Cannot find ref frame %s", ref_frames[m].c_str());
                continue;
            }

            Eigen::Matrix4d T_ref_src = T_ref_srcs[m];
            Eigen::Matrix4d T_ref_c1 = ref_poses[ref_frames[m]];
            // Eigen::Matrix4d T_src_c1 = T_src_ref * T_ref_c1;
            Eigen::Matrix4d T_src_c0 = src_poses[src_frames[m]];
            Eigen::Matrix4d T_ref_c0 = T_ref_src * T_src_c0;


            // debug. Render the gt camera poses
            // camera_pose_pnp_true.add_pose(T_src_cam_gt.block<3,1>(0,3), 
            //                         Eigen::Quaterniond(T_src_cam_gt.block<3,3>(0,0)));

            // second norm error
            Eigen::Matrix4d T_src_c1_gt = T_gt_ref_src.inverse() * T_ref_c1;
            Eigen::Matrix4d T_ref_c0_gt = T_gt_ref_src * T_src_c0;
            // double ate_error = (T_src_c1 - T_src_c1_gt).block<3,1>(0,3).norm();
            double ate_error = (T_ref_c0 - T_ref_c0_gt).block<3,1>(0,3).norm();

            Eigen::Matrix4d viz_pose = T_ref_c0; // T_src_c1 or T_ref_c0

            if(ate_error>ate_threshold){
                camera_pose_pnp_false.add_pose(viz_pose.block<3,1>(0,3), 
                                    Eigen::Quaterniond(viz_pose.block<3,3>(0,0)));
                loop_tp_masks.push_back(false);
                msg<<src_frames[m] <<" ";
                count_fp++;
            }
            else{
                camera_pose_pnp_true.add_pose(viz_pose.block<3,1>(0,3), 
                                        Eigen::Quaterniond(viz_pose.block<3,3>(0,0)));
                loop_tp_masks.push_back(true);
                count_true++;
            }
            
            src_positions.push_back(T_src_c0.block<3,1>(0,3));
            ref_positions.push_back(T_src_c1_gt.block<3,1>(0,3) + viz.t_local_remote[ref_name]);

        }

        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
        ros::spinOnce();
        ros::Duration(0.2).sleep();
        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
                                        


        msg<<std::endl;
        std::cout<<msg.str();
        ROS_WARN("Render %d predicted camera poses. %d tp camera poses, %d fp camera poses", 
                M, count_true, count_fp);

    }
    else if (mode==2){ // Seperate pose graph
        std::cout<<"No loop pose graph. Render sequence full pose graph."<<std::endl;

        if(select_frame_name.size()>1){
            // global_src_pcd->Clear();
            // global_src_pcd = open3d::io::CreatePointCloudFromFile(pose_graph_folder+"/"+select_frame_name+"_src.ply");
            // ROS_WARN("Override global map from %s", (pose_graph_folder+"/"+select_frame_name+"_src.ply").c_str());
            ROS_WARN("Render select frame pose %s", select_frame_name.c_str());
            Eigen::Matrix4d viz_pose;
            if(render_loops){
                std::vector<std::string> src_frames, ref_frames;
                std::vector<Eigen::Matrix4d> T_ref_srcs;
                read_our_registration(pose_graph_folder, src_frames, ref_frames, T_ref_srcs);
                int M = ref_frames.size();
                for (int m=0;m<M;m++){
                    if(src_frames[m]==select_frame_name){
                        T_ref_src_pred = T_ref_srcs[m];
                        break;
                    }
                }

                viz_pose = T_ref_src_pred * src_poses[select_frame_name];
                global_src_pcd->Transform(T_ref_src_pred);
            }
            else {
                if(src_poses.find(select_frame_name)==src_poses.end()){
                    ROS_WARN("Cannot find select frame %s", select_frame_name.c_str());
                }
                viz_pose = src_poses[select_frame_name];
            }
            Eigen::Vector3d p = viz_pose.block<3,1>(0,3);
            Eigen::Quaterniond q(viz_pose.block<3,3>(0,0));
            camera_pose_viz_Src.add_pose(p, q);
        }
        else{
            int frame_gap = 10;
            int idx = 0;            
            for(const auto &pose:src_poses){
                if(idx%frame_gap!=0){
                    idx++;
                    continue;
                }
                Eigen::Matrix4d viz_pose = T_gt_ref_src * pose.second;
                Eigen::Vector3d p = viz_pose.block<3,1>(0,3);
                Eigen::Quaterniond q(viz_pose.block<3,3>(0,0));
                camera_pose_viz_Src.add_pose(p, q);

                idx++;
            }
        }
    }
    else if (mode==10){ // HLoc PnP pose graph.
        std::cout<<"Render loop pose graph from file: "<<pose_graph_folder<<std::endl;
        std::unordered_map<std::string, Eigen::Matrix4d> src_frames_poses, ref_frames_poses;
        std::vector<LoopPair> loop_pairs;
        std::vector<Eigen::Matrix4d> loop_transformations;
        std::vector<Eigen::Vector3d> src_positions, ref_positions;
        std::vector<bool> loop_tp_masks;

        T_gt_ref_src.setIdentity();
        if(gt_file.size()>0){
            read_pose_file(gt_file, T_gt_ref_src); // gt src->ref
            Eigen::Quaterniond q_gt(T_gt_ref_src.block<3,3>(0,0));
            std::cout<<"Load ground truth transformation: "
                    <<T_gt_ref_src(0,3)<<" "<<T_gt_ref_src(1,3)<<" "<<T_gt_ref_src(2,3)<<" "
                    <<q_gt.x()<<" "<<q_gt.y()<<" "<<q_gt.z()<<" "<<q_gt.w()<<" "
                    <<std::endl;

            Eigen::Matrix4d T_inv = T_gt_ref_src.inverse();
            Eigen::Quaterniond q_inv(T_inv.block<3,3>(0,0));
            std::cout<<"Inverse ground truth transformation: "
                    <<T_inv(0,3)<<" "<<T_inv(1,3)<<" "<<T_inv(2,3)<<" "
                    <<q_inv.x()<<" "<<q_inv.y()<<" "<<q_inv.z()<<" "<<q_inv.w()<<" "
                    <<std::endl;
        }

        bool read_ret = read_frames_poses(pose_graph_folder+"/"+frame_file_name, src_frames_poses);
        if(!read_ret){
            return -1;
        }

        for(const auto &frame_pose:src_frames_poses){
            Eigen::Matrix4d T_gt_ref_cam;
            if(render_loops) T_gt_ref_cam = frame_pose.second;
            else T_gt_ref_cam = T_gt_ref_src.inverse() * frame_pose.second;

            camera_pose_viz_Src.add_pose(T_gt_ref_cam.block<3,1>(0,3), 
                                        Eigen::Quaterniond(T_gt_ref_cam.block<3,3>(0,0)));
        }        

        // read_loop
        if(render_loops){
            read_frames_poses(pose_graph_folder+"/ref_poses.txt", ref_frames_poses);
            bool read_loop_ret = read_loop_transformations(pose_graph_folder+"/loop_transformations.txt", loop_pairs, loop_transformations);
            assert(read_loop_ret);
        }    

        int count_tp = 0;
        int count_fp = 0;
        for (int loop_id=0;loop_id<loop_pairs.size();loop_id++){
            LoopPair loop_pair = loop_pairs[loop_id];
            std::string src_frame = loop_pair.first;
            std::string ref_frame = loop_pair.second;
            if(src_frames_poses.find(src_frame)==src_frames_poses.end()||
                ref_frames_poses.find(ref_frame)==ref_frames_poses.end()){
                ROS_WARN("Cannot find frame %s, %s", src_frame.c_str(), ref_frame.c_str());
                continue;
            }
            Eigen::Matrix4d T_src_c0 = src_frames_poses[src_frame]; // c0 if from src agent
            Eigen::Matrix4d T_ref_c1 = T_src_ref_viz * ref_frames_poses[ref_frame]; // c1 is from ref agent
            Eigen::Matrix4d T_pred_c0_c1 = loop_transformations[loop_id].inverse();
            Eigen::Matrix4d T_gt_src_c1 = T_gt_ref_src.inverse()  * ref_frames_poses[ref_frame]; // T_gt_src_ref * T_ref_c1


            Eigen::Matrix4d T_gt_ref_c0 = T_gt_ref_src * T_src_c0;

            Eigen::Vector3d viz_ref_pose = T_gt_src_c1.block<3,1>(0,3) + viz.t_local_remote[ref_name];
            src_positions.push_back(T_src_c0.block<3,1>(0,3));
            ref_positions.push_back(viz_ref_pose);

            // predicted camera pose by PnP
            Eigen::Matrix4d T_pred_ref_c0 = T_ref_c1 * T_pred_c0_c1.inverse();

            Eigen::Matrix4d T_pred_src_c1 = T_src_c0 * T_pred_c0_c1;
            double ate_c1 = (T_pred_src_c1 - T_gt_src_c1).block<3,1>(0,3).norm();
            // Eigen::Matrix4d T_pred_ref_c0 = src_frames_poses[src_frame];

            Eigen::Matrix4d viz_pose = T_pred_ref_c0;
            // Eigen::Matrix4d viz_pose = T_pred_src_c1;

            if(ate_c1<ate_threshold){ // true positive
                camera_pose_pnp_true.add_pose(viz_pose.block<3,1>(0,3), 
                                    Eigen::Quaterniond(viz_pose.block<3,3>(0,0)));
                loop_tp_masks.push_back(true);
                count_tp++;
            }
            else{ // false positive
                camera_pose_pnp_false.add_pose(viz_pose.block<3,1>(0,3), 
                                    Eigen::Quaterniond(viz_pose.block<3,3>(0,0)));
                loop_tp_masks.push_back(false);
                count_fp++;
            }

            // generate tp mask on loop edges
            // if(T_gt_ref_src.norm()>0){
            //     Eigen::Vector3d t_error = T_gt_ref_c0.block<3,1>(0,3) - ref_frames_poses[ref_frame].block<3,1>(0,3);
            //     if(t_error.norm()<ate_threshold) loop_tp_masks.push_back(true);
            //     else loop_tp_masks.push_back(false);
            // }
            // else loop_tp_masks.push_back(true);

        }

        camera_pose_pnp_true.publish_by(tp_camera_pose_pub, header_msg);
        camera_pose_pnp_false.publish_by(tp_camera_pose_pub, header_msg);
        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
                                        
        ros::spinOnce();
        ros::Duration(0.2).sleep();
        camera_pose_pnp_true.publish_by(tp_camera_pose_pub, header_msg);
        camera_pose_pnp_false.publish_by(tp_camera_pose_pub, header_msg);
        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
        if(render_loops)
            ROS_WARN("Render %d PnP optimized camera poses. %d tp, %d fp", loop_pairs.size(), count_tp, count_fp);
        std::cout<<"Publish "<<src_positions.size()<<" loop edges"<<std::endl;
    }
    else if (mode==11){ // HLoc PGO
        std::cout<<"Render loop pose graph from file: "<<pose_graph_folder<<std::endl;
        std::unordered_map<std::string, Eigen::Matrix4d> src_frames_poses, ref_frames_poses;
        std::vector<LoopPair> loop_pairs;
        std::vector<Eigen::Matrix4d> reg_transformations;
        std::vector<Eigen::Vector3d> src_positions, ref_positions;
        std::vector<bool> loop_tp_masks;

        bool read_ret = read_frames_poses(pose_graph_folder+"/"+frame_file_name, src_frames_poses);
        if(!read_ret) return -1;

        // Gt
        T_gt_ref_src.setIdentity();
        if(gt_file.size()>0) read_pose_file(gt_file, T_gt_ref_src); // gt src->ref
        
        // read_loop
        if(render_loops){
            read_frames_poses(pose_graph_folder+"/ref_poses.txt", ref_frames_poses);
            // read_entire_camera_poses(ref_scene_dir, ref_frames_poses);

            ROS_WARN("Load %ld ref poses from %s", ref_frames_poses.size(),
                                                (pose_graph_folder+"/ref_poses.txt").c_str());
            bool read_loop_ret = read_loop_transformations(pose_graph_folder+"/summary_pose_average.txt", loop_pairs, reg_transformations);
            assert(read_loop_ret);
        }

        for(const auto &frame_pose:src_frames_poses){
            Eigen::Matrix4d T_gt_ref_cam;
            if(render_loops) T_gt_ref_cam = frame_pose.second;
            else T_gt_ref_cam = T_gt_ref_src.inverse() * frame_pose.second;

            camera_pose_viz_Src.add_pose(T_gt_ref_cam.block<3,1>(0,3), 
                                        Eigen::Quaterniond(T_gt_ref_cam.block<3,3>(0,0)));
        }    

        //
        int pgo_count = 0;
        int count_tp = 0;
        int count_fp = 0;
        for (int loop_id=0;loop_id<loop_pairs.size();loop_id++){
            LoopPair loop_pair = loop_pairs[loop_id];
            std::string src_frame = loop_pair.first;
            std::string ref_frame = loop_pair.second;
            if(src_frames_poses.find(src_frame)==src_frames_poses.end()){
                ROS_WARN("Cannot find src frame %s", src_frame.c_str());
                continue;
            }
            if(ref_frames_poses.find(ref_frame)==ref_frames_poses.end()){
                ROS_WARN("Cannot find ref frame %s", ref_frame.c_str());
                continue;
            }

            Eigen::Matrix4d T_src_c0 = src_frames_poses[src_frame]; // c0 if from src agent
            Eigen::Matrix4d T_ref_c1 = ref_frames_poses[ref_frame]; // c1 is from ref agent
            Eigen::Matrix4d T_pred_ref_src = reg_transformations[loop_id];
            Eigen::Matrix4d T_ref_c0 = T_pred_ref_src * T_src_c0;
            Eigen::Matrix4d T_gt_ref_c0 = T_gt_ref_src * T_src_c0;
            Eigen::Matrix4d T_gt_src_c1 = T_gt_ref_src.inverse() * ref_frames_poses[ref_frame];

            double ate_c1 = (T_ref_c0 - T_gt_ref_c0).block<3,1>(0,3).norm();
            // Eigen::Matrix4d viz_pose = T_ref_c0;
            Eigen::Matrix4d viz_pose = T_gt_src_c1;

            src_positions.push_back(T_src_c0.block<3,1>(0,3));
            ref_positions.push_back(T_gt_src_c1.block<3,1>(0,3) + viz.t_local_remote[ref_name]);
            // std::cout<<ref_frame<<": "<<T_ref_c1.block<3,1>(0,3).transpose()<<std::endl;

            if(ate_c1<ate_threshold){ // true positive
                camera_pose_pnp_true.add_pose(viz_pose.block<3,1>(0,3), 
                                    Eigen::Quaterniond(viz_pose.block<3,3>(0,0)));
                loop_tp_masks.push_back(true);
                count_tp++;
            }
            else{ // false positive
                camera_pose_pnp_false.add_pose(viz_pose.block<3,1>(0,3), 
                                    Eigen::Quaterniond(viz_pose.block<3,3>(0,0)));
                loop_tp_masks.push_back(false);
                count_fp++;
            }
            pgo_count++;

        }

        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
        ros::spinOnce();
        ros::Duration(0.2).sleep();
        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
        if(render_loops)
            ROS_WARN("Render %d PGO optimized camera poses. %d tp, %d fp", pgo_count, count_tp, count_fp);

    }
    else if (mode==15){
        ROS_WARN("Render Hydra loop pose graph from: %s", pose_graph_folder.c_str());
        std::vector<std::string> ref_frames;
        std::unordered_map<std::string, Eigen::Matrix4d> ref_poses;
        std::vector<fmfusion::LoopPair> loop_pairs;
        
        read_entire_camera_poses(ref_scene_dir, ref_poses);
        fmfusion::IO::read_loop_pairs(pose_graph_folder+"/"+loop_filename, loop_pairs, loop_correct_masks);
        std::cout<<"Load "<<loop_pairs.size()<<" loop pairs from "<<loop_filename<<std::endl;

        for(const auto&loop_pair: loop_pairs){
            std::string src_frame = loop_pair.first;
            std::string ref_frame = loop_pair.second;
            if(ref_poses.find(ref_frame)==ref_poses.end()){
                ROS_WARN("Cannot find ref frame %s", ref_frame.c_str());
                continue;
            }
            Eigen::Matrix4d src_pose = src_poses[src_frame];
            Eigen::Vector3d aligned_src_position = (T_gt_ref_src * src_pose).block<3,1>(0,3);

            loop_src_point.push_back(aligned_src_position);
            loop_ref_point.push_back(ref_poses[ref_frame].block<3,1>(0,3) + viz.t_local_remote[ref_name]);
        }
        std::cout<<"Update "<<loop_src_point.size()<<" valid loop edges"<<std::endl;

    }
    else{
        ROS_ERROR("%s Just render global map", src_name.c_str());
    }

    // Pub
    ROS_INFO("---------- Visualization ----------");    
    for (int i=0;i<iter_num ;i++){
        camera_pose_viz_Src.publish_by(src_camera_pose_pub, header_msg);        
        camera_pose_pnp_true.publish_by(tp_camera_pose_pub, header_msg);
        camera_pose_pnp_false.publish_by(fp_camera_pose_pub, header_msg);
        Visualization::render_point_cloud(global_src_pcd, 
                                            viz.src_global_map, 
                                            src_name);
        std::cout<<src_name <<" publish "<< global_src_pcd->points_.size()
                                <<" global points"<<std::endl;

        if (loop_src_point.size()>0){
            Visualization::correspondences(loop_src_point, loop_ref_point, 
                                            loop_edge_pub, src_name, loop_correct_masks);
        }

        if(render_trajectory){
            std::vector<Eigen::Matrix4d> src_poses_viz;
            read_entire_camera_poses(src_scene_dir, src_poses_viz);
            std::cout<<"Generate "<<src_poses_viz.size()<<" poses "
                        <<"from "<<src_scene_dir<<std::endl;
        
            //
            nav_msgs::Path src_path_msg;
            Visualization::render_path(src_poses_viz,
                                        T_gt_ref_src,
                                        src_name,
                                        src_path_msg,
                                        viz.path);
        }

        Visualization::correspondences(src_centroids_m, ref_centroids_m, 
                                        viz.instance_match, src_name, 
                                        node_tp_masks, T_src_ref_viz);

        if(src_centroids.size()>0) 
            Visualization::instance_centroids(src_centroids, viz.src_centroids, 
                                            src_name, 
                                            viz.param.centroid_size,
                                            viz.param.centroid_color);

        if(src_instance_annotations.size()>0 && !render_superpoint) 
            Visualization::node_annotation(src_centroids, 
                                        src_instance_annotations, 
                                        viz.node_annotation, 
                                        src_name,
                                        viz.param.annotation_size,
                                        viz.param.annotation_voffset,
                                        viz.param.annotation_color); 

        if(mode==0 && dense_corr) Visualization::correspondences(src_corrs_points, ref_corrs_points, 
                                        viz.point_match, src_name, 
                                        corr_mask, T_src_ref_viz);


        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }

    return 0;

}
