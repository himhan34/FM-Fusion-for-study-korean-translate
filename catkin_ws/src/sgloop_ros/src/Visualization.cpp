#include "Visualization.h"

namespace Visualization
{

    bool render_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, ros::Publisher pub, std::string frame_id)
    {
        // Publish point cloud
        sensor_msgs::PointCloud2 msg;
        open3d_conversions::open3dToRos(*pcd, msg, frame_id);
        pub.publish(msg);
        std::cout<<"Published point cloud to "<<frame_id<<std::endl;

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

    bool instance_match(const std::vector<Eigen::Vector3d> &src_centroids,
                        const std::vector<Eigen::Vector3d> &ref_centroids,
                        ros::Publisher pub, std::string frame_id, std::vector<bool> pred_masks)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
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
            p2.x = ref_centroids[i].x();
            p2.y = ref_centroids[i].y();
            p2.z = ref_centroids[i].z();
            marker.points.push_back(p1);
            marker.points.push_back(p2);
            std_msgs::ColorRGBA line_color;
            line_color.a = 1;
            // line_color.r = 1;
            if(!pred_masks.empty()){
                if(pred_masks[i]) line_color.g = 1;
                else line_color.r = 1;
            }
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
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
            msg->header.frame_id = frame_id;
            pub.publish(msg);
            // std::cout<<"Published image to "<<frame_id<<std::endl;
            return true;
        }
    }

}