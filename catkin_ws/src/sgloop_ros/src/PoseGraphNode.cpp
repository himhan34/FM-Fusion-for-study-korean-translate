#include <sstream>
#include <unordered_map>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "open3d_conversions/open3d_conversions.h"

#include "tools/IO.h"

#include "Visualization.h"
#include "utility/CameraPoseVisualization.h"

typedef std::pair<std::string, std::string> LoopPair;

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
        std::cout<<frame_name<<std::endl;
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
    std::string loop_file;
    float camera_color[4];

    nh_private.getParam("src_name", src_name);
    nh_private.getParam("ref_name", ref_name);
    nh_private.getParam("src_scene_dir", src_scene_dir);
    camera_color[0] = nh_private.param("camera_marker/r", 1.0);
    camera_color[1] = nh_private.param("camera_marker/g", 0.0);
    camera_color[2] = nh_private.param("camera_marker/b", 0.0);
    camera_color[3] = nh_private.param("camera_marker/a", 1.0);
    float camera_scale = nh_private.param("camera_marker/scale", 0.1);
    float camera_line_width = nh_private.param("camera_marker/line_width", 0.01);
    float ate_threshold = nh_private.param("ate_threshold", 0.5);
    bool map_only = nh_private.param("map_only", false);
    std::string pose_graph_folder = nh_private.param("pose_graph_folder", std::string(""));
    std::string gt_file = nh_private.param("gt_file", std::string(""));
    std::string frame_file_name = nh_private.param("frame_file_name", std::string("src_poses.txt"));

    // 
    Eigen::Vector3d t_src_ref;
    Eigen::Matrix4d T_src_ref_viz = Eigen::Matrix4d::Identity(); // src->ref incorporated for vizualization

    t_src_ref[0] = nh.param("br/"+ref_name+"/x", 0.0);
    t_src_ref[1] = nh.param("br/"+ref_name+"/y", 0.0);
    t_src_ref[2] = nh.param("br/"+ref_name+"/z", 0.0);
    float yaw_src_ref = nh.param("br/"+ref_name+"/yaw", 0.0);
    T_src_ref_viz.block<3,1>(0,3) = t_src_ref;
    T_src_ref_viz.block<3,3>(0,0) = Eigen::AngleAxisd(yaw_src_ref, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Init viz tool
    Visualization::Visualizer viz(nh, nh_private);
    CameraPoseVisualization camera_pose_viz_Src(camera_color[0], camera_color[1], camera_color[2], camera_color[3]);
    camera_pose_viz_Src.setScale(camera_scale);
    camera_pose_viz_Src.setLineWidth(camera_line_width);

    // Load map
    auto src_pcd = open3d::io::CreatePointCloudFromFile(src_scene_dir+"/mesh_o3d.ply");

    // Visualization
    std_msgs::Header header_msg;
    header_msg.frame_id = src_name;
    header_msg.stamp = ros::Time::now();

    // Load pose graph
    if(map_only){
        std::cout<<"Display map only!\n";
    }
    else if (pose_graph_folder.size()<1){
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
    else{
        std::cout<<"Render loop pose graph from file: "<<pose_graph_folder<<std::endl;
        std::unordered_map<std::string, Eigen::Matrix4d> src_frames_poses, ref_frames_poses;
        std::vector<LoopPair> loop_pairs;
        std::vector<Eigen::Matrix4d> loop_transformations;
        std::vector<Eigen::Vector3d> src_positions, ref_positions;
        Eigen::Matrix4d T_gt_ref_src;        

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
    
        // for (const LoopPair &loop_pair:loop_pairs){ // Loops
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
    for (int i=0;i<2;i++){
        camera_pose_viz_Src.publish_by(src_camera_pose_pub, header_msg);        
        Visualization::render_point_cloud(src_pcd, viz.src_global_map, src_name);
        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }

    return 0;

}
