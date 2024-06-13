#include "Visualization.h"

namespace Visualization
{
    Visualizer::Visualizer(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
    {
        // Ref graph
        ref_graph = nh_private.advertise<sensor_msgs::PointCloud2>("ref/instance_map", 1000);
        ref_centroids = nh_private.advertise<visualization_msgs::Marker>("ref/centroids", 1000);
        // ref_edges = nh_private.advertise<visualization_msgs::Marker>("ref/edges", 1000);

        // Local graph
        src_graph = nh_private.advertise<sensor_msgs::PointCloud2>("instance_map", 1000);
        src_centroids = nh_private.advertise<visualization_msgs::Marker>("centroids", 1000);
        src_edges = nh_private.advertise<visualization_msgs::Marker>("edges", 1000);

        // Loop
        instance_match = nh_private.advertise<visualization_msgs::Marker>("instance_match", 1000);
        point_match = nh_private.advertise<visualization_msgs::Marker>("point_match", 1000);
        src_map_aligned = nh_private.advertise<sensor_msgs::PointCloud2>("aligned_map", 1000); //aligned and render in reference frame

        rgb_image = nh_private.advertise<sensor_msgs::Image>("rgb_image", 1000);
        camera_pose = nh_private.advertise<nav_msgs::Odometry>("camera_pose", 1000);
        path = nh_private.advertise<nav_msgs::Path>("path", 1000);


        path_msg.header.stamp = ros::Time::now();

        // Global viz setting
        param.edge_width = nh.param("viz/edge_width", 0.02);
        param.edge_color[0] = nh.param("viz/edge_color/r", 0.0);
        param.edge_color[1] = nh.param("viz/edge_color/g", 0.0);
        param.edge_color[2] = nh.param("viz/edge_color/b", 1.0);
        param.centroid_size = nh.param("viz/centroid_size", 0.1);

        // Agents relative transform
        local_frame_offset[0] = nh_private.param("viz/local_frame_offset/x", 0.0);
        local_frame_offset[1] = nh_private.param("viz/local_frame_offset/y", 0.0);
        local_frame_offset[2] = nh_private.param("viz/local_frame_offset/z", 0.0);
        std::cout<<"local_frame_offset: "<<local_frame_offset[0]<<" "<<local_frame_offset[1]<<" "<<local_frame_offset[2]<<std::endl;
    }

    bool render_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, ros::Publisher pub, std::string frame_id)
    {
        // Publish point cloud
        sensor_msgs::PointCloud2 msg;
        open3d_conversions::open3dToRos(*pcd, msg, frame_id);
        pub.publish(msg);

        return true;
    }

    bool inter_graph_edges(const std::vector<Eigen::Vector3d> &centroids,
                            const std::vector<std::pair<int,int>> &edges,
                            ros::Publisher pub,
                            float width, 
                            std::array<float,3> color,
                            std::string frame_id)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "inter_graph_edges";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = width; // line width
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = 1.0;

        for (int i=0;i<edges.size();i++){
            geometry_msgs::Point p1, p2;
            p1.x = centroids[edges[i].first].x();
            p1.y = centroids[edges[i].first].y();
            p1.z = centroids[edges[i].first].z();
            p2.x = centroids[edges[i].second].x();
            p2.y = centroids[edges[i].second].y();
            p2.z = centroids[edges[i].second].z();
            marker.points.push_back(p1);
            marker.points.push_back(p2);
        }

        pub.publish(marker);

        return true;
    }

    bool correspondences(const std::vector<Eigen::Vector3d> &src_centroids,
                        const std::vector<Eigen::Vector3d> &ref_centroids,
                        ros::Publisher pub, 
                        std::string src_frame_id, 
                        std::vector<bool> pred_masks,
                        std::array<float,3> src_frame_offset)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = src_frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "instance_match";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.02; // line width
        marker.color.a = 1.0;

        int n = src_centroids.size();
        assert(n==ref_centroids.size());

        for (int i=0;i<n;i++){
            geometry_msgs::Point p1, p2;
            p1.x = src_centroids[i].x();
            p1.y = src_centroids[i].y();
            p1.z = src_centroids[i].z();
            p2.x = ref_centroids[i].x()+ src_frame_offset[0];
            p2.y = ref_centroids[i].y()+ src_frame_offset[1];
            p2.z = ref_centroids[i].z()+ src_frame_offset[2];
            marker.points.push_back(p1);
            marker.points.push_back(p2);
            std_msgs::ColorRGBA line_color;
            line_color.a = 1;
            // line_color.r = 1;
            if(!pred_masks.empty()){
                if(pred_masks[i]) line_color.g = 1;
                else line_color.r = 1;
            }
            else line_color.b = 1;
            marker.colors.push_back(line_color);
            marker.colors.push_back(line_color);
        }

        pub.publish(marker);

        return true;

    }

    bool instance_centroids(const std::vector<Eigen::Vector3d> &centroids,
                            ros::Publisher pub, 
                            std::string frame_id, 
                            float scale,
                            std::array<float,3> color)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "instance_centroid";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = scale;
        marker.scale.y = scale;
        marker.scale.z = scale;
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = 1.0;

        for (int i=0;i<centroids.size();i++){
            geometry_msgs::Point p;
            p.x = centroids[i].x();
            p.y = centroids[i].y();
            p.z = centroids[i].z();
            marker.points.push_back(p);
        }

        pub.publish(marker);
        return true;
    }

    bool render_image(const cv::Mat &image, ros::Publisher pub, std::string frame_id)
    {
        if(pub.getNumSubscribers()==0) return false;
        else{ // Publish image
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", image).toImageMsg();
            msg->header.frame_id = frame_id;
            pub.publish(msg);
            return true;
        }
    }

    bool render_camera_pose(const Eigen::Matrix4d &pose, ros::Publisher pub, std::string frame_id, int sequence_id)
    {
        nav_msgs::Odometry msg;
        msg.header.frame_id = frame_id;
        msg.header.seq = sequence_id;
        msg.header.stamp = ros::Time::now();
        msg.child_frame_id = "camera_pose";
        msg.pose.pose.position.x = pose(0,3);
        msg.pose.pose.position.y = pose(1,3);
        msg.pose.pose.position.z = pose(2,3);
        Eigen::Quaterniond q(pose.block<3,3>(0,0));
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.pose.orientation.w = q.w();
        pub.publish(msg);
        return true;
        
    }

    bool render_path (const Eigen::Matrix4d &poses, nav_msgs::Path &path_msg,ros::Publisher pub,std::string frame_id, int sequence_id)
    {
        path_msg.header.frame_id = frame_id;

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = frame_id;
        pose_stamped.header.seq = sequence_id;
        pose_stamped.header.stamp = ros::Time::now();
        pose_stamped.pose.position.x = poses(0,3);
        pose_stamped.pose.position.y = poses(1,3);
        pose_stamped.pose.position.z = poses(2,3);
        Eigen::Quaterniond q(poses.block<3,3>(0,0));
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();
        path_msg.poses.push_back(pose_stamped);
        pub.publish(path_msg);
        return true;
    }

                

}