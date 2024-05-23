#include "unordered_map"
#include "ros/ros.h"

#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "open3d_conversions/open3d_conversions.h"

namespace Visualization
{

    bool render_point_cloud(const std::shared_ptr<open3d::geometry::PointCloud> &pcd, ros::Publisher pub, std::string frame_id="world");

    bool instance_match(const std::vector<Eigen::Vector3d> &src_centroids,
                        const std::vector<Eigen::Vector3d> &ref_centroids,
                        ros::Publisher pub, std::string frame_id="world");

    bool instance_centroids(const std::vector<Eigen::Vector3d> &centroids,
                            ros::Publisher pub, 
                            std::string frame_id="world", 
                            std::array<uint8_t,3> color={255,0,0});

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