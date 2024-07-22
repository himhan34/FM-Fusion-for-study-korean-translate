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
        std::string frame_name = open3d::utility::filesystem::GetFileNameWithoutDirectory(src_rgbds[i].first);
        // std::cout<<frame_name<<std::endl;
        src_poses_map[src_rgbds[i].first] = src_poses[i];
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
                        // std::vector<Eigen::Matrix4d> &ref_poses)
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

    ros::init(argc, argv, "LoopNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    ros::Publisher src_camera_pose_pub = nh_private.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
    ros::Publisher pnp_camera_pose_pub = nh_private.advertise<visualization_msgs::MarkerArray>("pnp_pose_graph", 1000);
    ros::Publisher loop_edge_pub = nh_private.advertise<visualization_msgs::Marker>("loop_edge", 1000);
    std::string src_name, ref_name;
    std::string src_scene_dir;
    std::string scene_result_folder;
    std::string loop_file;
    float camera_color[4];

    // Params
    nh_private.getParam("src_name", src_name);
    nh_private.getParam("ref_name", ref_name);
    nh_private.getParam("src_scene_dir", src_scene_dir);
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
    std::string map_name = nh_private.param("map_name", std::string("mesh_o3d.ply"));
    std::string pose_graph_folder = nh_private.param("pose_graph_folder", std::string(""));
    std::string gt_file = nh_private.param("gt_file", std::string(""));
    std::string frame_file_name = nh_private.param("frame_file_name", std::string("src_poses.txt"));
    float corr_sampling_ratio = nh_private.param("corr_sampling_ratio", 0.1);

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
    Visualization::Visualizer viz(nh, nh_private);
    CameraPoseVisualization camera_pose_viz_Src(camera_color[0], camera_color[1], camera_color[2], camera_color[3]);
    camera_pose_viz_Src.setScale(camera_scale);
    camera_pose_viz_Src.setLineWidth(camera_line_width);
    std::vector<Eigen::Vector3d> src_centroids_m, ref_centroids_m;
    std::vector<Eigen::Vector3d> src_corrs_points, ref_corrs_points;
    std::vector<bool> corr_mask;
    std::vector<std::pair<uint32_t,uint32_t>> node_pairs;
    std::vector<bool> node_tp_masks;

    // Load map
    Eigen::Matrix4d T_gt_ref_src;        
    std::vector<Eigen::Vector3d> src_centroids;
    std::shared_ptr<open3d::geometry::PointCloud> global_src_pcd;
    global_src_pcd = open3d::io::CreatePointCloudFromFile(src_scene_dir+"/"+map_name);
    std::cout<<"Global map has "<<global_src_pcd->points_.size()<<" points, "
            <<"It is loaded  "<<src_scene_dir+"/"+map_name<<std::endl;

    // Visualization
    std_msgs::Header header_msg;
    header_msg.frame_id = src_name;
    header_msg.stamp = ros::Time::now();

    // Load
    if(map_only){ // matches only
        using namespace fmfusion;
        std::vector<InstanceId> src_names;
        std::vector<InstancePtr> src_instances;   
        std::string config_file;

        bool set_cfg = nh_private.getParam("cfg_file", config_file);
        assert(set_cfg);
        Config *sg_config = utility::create_scene_graph_config(config_file, true);
        if(gt_file.size()>0){
            read_pose_file(gt_file, T_gt_ref_src); // gt src->ref
        }
        else T_gt_ref_src.setIdentity();
        std::cout<<"Load scene relative pose from "<<gt_file<<std::endl;

        auto src_mapping = std::make_shared<SemanticMapping>(SemanticMapping(sg_config->mapping_cfg,sg_config->instance_cfg));
        src_mapping->load(src_scene_dir);
        src_mapping->extract_bounding_boxes();
        src_mapping->export_instances(src_names,src_instances);


        auto src_graph = std::make_shared<Graph>(sg_config->graph);
        src_graph->initialize(src_instances);
        src_graph->construct_edges();
        src_graph->construct_triplets();

        global_src_pcd->Clear();
        src_centroids = src_graph->get_centroids();
        global_src_pcd = src_graph->extract_global_cloud();
        for(auto &center:src_centroids){
            center = T_gt_ref_src.block<3,3>(0,0) * center + T_gt_ref_src.block<3,1>(0,3);
        }

        global_src_pcd->Transform(T_gt_ref_src);

        if(scene_result_folder.size()>0){ // Load node match results
            std::cout<<"Load match results from "<<scene_result_folder<<std::endl;
            fmfusion::IO::load_node_matches(scene_result_folder+"/node_matches.txt",
                                            node_pairs, node_tp_masks,
                                            src_centroids_m, ref_centroids_m);
            if(dense_corr){
                auto corr_src_pcd = open3d::io::CreatePointCloudFromFile(scene_result_folder+"/corr_src.ply");
                auto corr_ref_pcd = open3d::io::CreatePointCloudFromFile(scene_result_folder+"/corr_ref.ply");
                auto src_corrs_points_ = corr_src_pcd->points_;
                auto ref_corrs_points_ = corr_ref_pcd->points_;
                std::vector<bool> corr_mask_;
                std::cout<<"Load "<<src_corrs_points_.size()<<" dense correspondences"<<std::endl;

                if(!fmfusion::IO::load_single_col_mask(scene_result_folder+"/corres_pos.txt",corr_mask_))
                {
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
    else if (pose_graph_folder.size()<1){ // Seperate pose graph
        std::cout<<"No loop pose graph. Render sequence full pose graph."<<std::endl;
        std::unordered_map<std::string, Eigen::Matrix4d> src_poses;
        bool read_ret = read_entire_camera_poses(src_scene_dir, src_poses);
        std::cout<<"Load "<<src_poses.size()<<" poses"<<std::endl;    

        for(const auto &pose:src_poses){
            Eigen::Vector3d p = pose.second.block<3,1>(0,3);
            Eigen::Quaterniond q(pose.second.block<3,3>(0,0));
            camera_pose_viz_Src.add_pose(p, q);
        }
    }
    else{ // Aligned pose graph
        std::cout<<"Render loop pose graph from file: "<<pose_graph_folder<<std::endl;
        std::unordered_map<std::string, Eigen::Matrix4d> src_frames_poses, ref_frames_poses;
        std::vector<LoopPair> loop_pairs;
        std::vector<Eigen::Matrix4d> loop_transformations;
        std::vector<Eigen::Vector3d> src_positions, ref_positions;

        CameraPoseVisualization camera_pose_pnp_true(0.0, 1.0, 0.0, 1.0);
        CameraPoseVisualization camera_pose_pnp_false(1.0, 0.0, 0.0, 1.0);
        camera_pose_pnp_true.setScale(camera_scale);
        camera_pose_pnp_true.setLineWidth(camera_line_width);
        camera_pose_pnp_false.setScale(camera_scale);
        camera_pose_pnp_false.setLineWidth(camera_line_width);

        T_gt_ref_src.setZero();
        std::vector<bool> loop_tp_masks;
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
        read_frames_poses(pose_graph_folder+"/ref_poses.txt", ref_frames_poses);

        // read_loop
        bool read_loop_ret = read_loop_transformations(pose_graph_folder+"/loop_transformations.txt", loop_pairs, loop_transformations);
        if(!read_loop_ret){
            return -1;
        }

        for(const auto &frame_pose:src_frames_poses){
            camera_pose_viz_Src.add_pose(frame_pose.second.block<3,1>(0,3), 
                                        Eigen::Quaterniond(frame_pose.second.block<3,3>(0,0)));
        }        
    
        for (int loop_id=0;loop_id<loop_pairs.size();loop_id++){
            LoopPair loop_pair = loop_pairs[loop_id];
            std::string src_frame = loop_pair.first;
            std::string ref_frame = loop_pair.second;
            Eigen::Matrix4d T_src_c0 = src_frames_poses[src_frame]; // c0 if from src agent
            Eigen::Matrix4d T_ref_c1 = T_src_ref_viz * ref_frames_poses[ref_frame]; // c1 is from ref agent
            Eigen::Matrix4d T_pred_c0_c1 = loop_transformations[loop_id].inverse();

            src_positions.push_back(T_src_c0.block<3,1>(0,3));
            ref_positions.push_back(T_ref_c1.block<3,1>(0,3));

            // predicted camera pose by PnP
            Eigen::Matrix4d T_pred_src_c1 = T_src_c0 * T_pred_c0_c1;
            Eigen::Matrix4d T_gt_src_c1 = T_gt_ref_src.inverse()  * ref_frames_poses[ref_frame]; // T_gt_src_ref * T_ref_c1
            double ate_c1 = (T_pred_src_c1 - T_gt_src_c1).block<3,1>(0,3).norm();
            // Eigen::Matrix4d T_pred_ref_c0 = src_frames_poses[src_frame];

            if(ate_c1<ate_threshold){ // true positive
                camera_pose_pnp_true.add_pose(T_pred_src_c1.block<3,1>(0,3), 
                                    Eigen::Quaterniond(T_pred_src_c1.block<3,3>(0,0)));
            }
            else{ // false positive
                camera_pose_pnp_false.add_pose(T_pred_src_c1.block<3,1>(0,3), 
                                    Eigen::Quaterniond(T_pred_src_c1.block<3,3>(0,0)));
            }

            // generate tp mask on loop edges
            if(T_gt_ref_src.norm()>0){
                Eigen::Matrix4d T_gt_ref_c0 = T_gt_ref_src * T_src_c0;
                Eigen::Vector3d t_error = T_gt_ref_c0.block<3,1>(0,3) - ref_frames_poses[ref_frame].block<3,1>(0,3);
                if(t_error.norm()<ate_threshold) loop_tp_masks.push_back(true);
                else loop_tp_masks.push_back(false);
            }
            else loop_tp_masks.push_back(true);

        }

        camera_pose_pnp_true.publish_by(pnp_camera_pose_pub, header_msg);
        camera_pose_pnp_false.publish_by(pnp_camera_pose_pub, header_msg);
        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
        ros::spinOnce();
        ros::Duration(0.2).sleep();
        camera_pose_pnp_true.publish_by(pnp_camera_pose_pub, header_msg);
        camera_pose_pnp_false.publish_by(pnp_camera_pose_pub, header_msg);
        Visualization::correspondences(src_positions, ref_positions, 
                                        loop_edge_pub, src_name, loop_tp_masks);
        std::cout<<"Publish "<<src_positions.size()<<" loop edges"<<std::endl;
    }

    // Pub    
    for (int i=0;i<iter_num ;i++){
        camera_pose_viz_Src.publish_by(src_camera_pose_pub, header_msg);        
        Visualization::render_point_cloud(global_src_pcd, viz.src_global_map, src_name);

        std::vector<Eigen::Matrix4d> src_poses_viz;
        read_entire_camera_poses(src_scene_dir, src_poses_viz);
        std::cout<<"Generate path from "<<src_poses_viz.size()<<" poses"<<std::endl;
    
        //
        nav_msgs::Path src_path_msg;
        Visualization::render_path(src_poses_viz,
                                    Eigen::Matrix4d::Identity(),
                                    src_name,
                                    src_path_msg,
                                    viz.path);

        Visualization::correspondences(src_centroids_m, ref_centroids_m, 
                                        viz.instance_match, src_name, 
                                        node_tp_masks, T_src_ref_viz);
        if(map_only) Visualization::instance_centroids(src_centroids, viz.src_centroids, 
                                                        src_name, 
                                                        viz.param.centroid_size,
                                                        viz.param.centroid_color);
        if(map_only && dense_corr) Visualization::correspondences(src_corrs_points, ref_corrs_points, 
                                        viz.point_match, src_name, 
                                        corr_mask, T_src_ref_viz);

        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }

    return 0;

}
