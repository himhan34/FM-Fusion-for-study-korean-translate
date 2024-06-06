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
                            std::array<uint8_t,3> color)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "instance_centroid";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
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

        // geometry_msgs::PointStamped points;
        // points.header.frame_id = frame_id;
        // points.header.stamp = ros::Time::now();
        // for (int i=0;i<centroids.size();i++){
        //     geometry_msgs::Point p;
        //     p.x = centroids[i].x();
        //     p.y = centroids[i].y();
        //     p.z = centroids[i].z();
        //     points.point = p;
        // }

        // pub.publish(points);


        return true;

    }


}