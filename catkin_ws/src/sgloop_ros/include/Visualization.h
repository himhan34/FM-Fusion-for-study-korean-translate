#include "unordered_map"
#include "ros/ros.h"

#include "std_msgs/ColorRGBA.h"
#include "geometry_msgs/PointStamped.h"
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

    bool render_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, ros::Publisher pub, std::string frame_id="world");

    bool inter_graph_edges(const std::vector<Eigen::Vector3d> &centroids,
                            const std::vector<std::pair<int,int>> &edges,
                            ros::Publisher pub,
                            float width=0.02,
                            std::array<float,3> color={0.0,0.0,1.0},
                            std::string frame_id="world");

    bool instance_match(const std::vector<Eigen::Vector3d> &src_centroids,
                        const std::vector<Eigen::Vector3d> &ref_centroids,
                        ros::Publisher pub, 
                        std::string frame_id="world",
                        std::vector<bool> pred_masks={});

    bool instance_centroids(const std::vector<Eigen::Vector3d> &centroids,
                            ros::Publisher pub, 
                            std::string frame_id="world", 
                            float scale=0.1,
                            std::array<float,3> color={0.0,0.0,1.0});

    bool render_image(const cv::Mat &image, ros::Publisher pub, std::string frame_id="world");

    // class Visualizer
    // {
    // public:
    //     Visualizer();
    //     ~Visualizer(){};

    //     void initialize(ros::NodeHandle &n, ros::NodeHandle &nh_private);

    //     bool render_point_cloud(const std::string &pub_name, const std::shared_ptr<open3d::geometry::PointCloud> &pcd);

    // private:
    //     std::string mode;

    //     ros::Publisher ref_graph, src_graph;
    //     // std::unordered_map<std::string, ros::Publisher> publishers;


    // };

}