#include "unordered_map"
#include "ros/ros.h"

#include "std_msgs/ColorRGBA.h"
#include "geometry_msgs/PointStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "visualization_msgs/Marker.h"

#include "cv_bridge/cv_bridge.h"
#include "open3d_conversions/open3d_conversions.h"

namespace Visualization
{
    struct VizParam{
        float edge_width = 0.02;
        std::array<float,3> edge_color;
        float centroid_size = 0.1;
        std::array<float,3> centroid_color = {0.0,0.0,0.0};
    }; 

    class Visualizer
    {
        public:
            Visualizer(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
            ~Visualizer(){};

        public:
            ros::Publisher ref_graph, src_graph;
            ros::Publisher ref_centroids, src_centroids;
            ros::Publisher ref_edges, src_edges;
            ros::Publisher instance_match, point_match;
            ros::Publisher src_map_aligned;

            ros::Publisher rgb_image;
            ros::Publisher camera_pose, path;

            nav_msgs::Path path_msg;
            VizParam param;
            std::array<float,3> local_frame_offset;

    };

    bool render_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, ros::Publisher pub, std::string frame_id="world");

    bool inter_graph_edges(const std::vector<Eigen::Vector3d> &centroids,
                            const std::vector<std::pair<int,int>> &edges,
                            ros::Publisher pub,
                            float width=0.02,
                            std::array<float,3> color={0.0,0.0,1.0},
                            std::string frame_id="world");

    bool correspondences(const std::vector<Eigen::Vector3d> &src_centroids,
                        const std::vector<Eigen::Vector3d> &ref_centroids,
                        ros::Publisher pub, 
                        std::string frame_id="world",
                        std::vector<bool> pred_masks={},
                        std::array<float,3> src_frame_offset={0.0,0.0,0.0});

    bool instance_centroids(const std::vector<Eigen::Vector3d> &centroids,
                            ros::Publisher pub, 
                            std::string frame_id="world", 
                            float scale=0.1,
                            std::array<float,3> color={0.0,0.0,1.0});

    bool render_image(const cv::Mat &image, ros::Publisher pub, std::string frame_id="world");

    bool render_camera_pose(const Eigen::Matrix4d &pose, ros::Publisher pub, 
                            std::string frame_id="world", int sequence_id = 0);

    bool render_path(const Eigen::Matrix4d &poses, 
                    nav_msgs::Path &path_msg,
                    ros::Publisher pub, std::string frame_id="world", int sequence_id = 0);
}