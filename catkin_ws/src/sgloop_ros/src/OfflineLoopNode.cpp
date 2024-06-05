#include <sstream>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"

#include "open3d_conversions/open3d_conversions.h"

#include "SceneGraph.h"
#include "tools/Tools.h"
#include "sgloop/Graph.h"
#include "sgloop/LoopDetector.h"
// #include "sgloop/ShapeEncoder.h"

#include "Visualization.h"

struct Publishers{
  ros::Publisher ref_graph, src_graph;
  ros::Publisher ref_centroids, src_centroids;
  ros::Publisher instance_match;
}mypubs;

struct GraphMsg{
  sensor_msgs::PointCloud2 global_cloud;
}ref_msgs, src_msgs;


int main(int argc, char **argv)
{
    using namespace fmfusion;
    ros::init(argc, argv, "LoopNode");
    ros::NodeHandle n;
    ros::NodeHandle nh_private("~");

    // Settings
    std::string config_file, weights_folder;
    std::string ref_scene_dir, src_scene_dir;

    bool set_cfg = nh_private.getParam("cfg_file", config_file);
    bool set_wegith_folder = nh_private.getParam("weights_folder", weights_folder);
    bool set_ref_scene = nh_private.getParam("ref_scene_dir", ref_scene_dir);
    bool set_src_scene = nh_private.getParam("src_scene_dir", src_scene_dir);
    std::string output_folder = nh_private.param("output_folder", std::string(""));
    std::string frame_id = nh_private.param("frame_id", std::string("world"));
    int visualization = nh_private.param("visualization", 0);

    // Publisher
    mypubs.ref_graph = nh_private.advertise<sensor_msgs::PointCloud2>("ref_graph", 1000);
    mypubs.src_graph = nh_private.advertise<sensor_msgs::PointCloud2>("src_graph", 1000);
    mypubs.ref_centroids = nh_private.advertise<visualization_msgs::Marker>("ref_centroids", 1000);
    mypubs.src_centroids = nh_private.advertise<visualization_msgs::Marker>("src_centroids", 1000);
    mypubs.instance_match = nh_private.advertise<visualization_msgs::Marker>("instance_match", 1000);

    assert(set_cfg && set_wegith_folder);
    assert(set_ref_scene && set_src_scene);

    std::cout<<"config file: "<<config_file<<std::endl;

    // Load Config
    GraphConfig graph_config;
    SgNetConfig loop_config;
    // ShapeEncoderConfig shape_config;
    Config *sg_config = utility::create_scene_graph_config(config_file, true);
    if (sg_config==nullptr){
        ROS_WARN("Failed to create scene graph config.");
        return 0;
    }

    // Load reconstrcuted maps
    auto ref_map = std::make_shared<SceneGraph>(SceneGraph(*sg_config));
    auto src_map = std::make_shared<SceneGraph>(SceneGraph(*sg_config));
    ref_map->load(ref_scene_dir);
    src_map->load(src_scene_dir);
    ref_map->extract_bounding_boxes();
    src_map->extract_bounding_boxes();

    std::vector<InstanceId> ref_names, src_names;
    std::vector<InstancePtr> ref_instances, src_instances;
    ref_map->export_instances(ref_names,ref_instances);
    src_map->export_instances(src_names,src_instances);

    // Construct explicit graph
    auto ref_graph = std::make_shared<Graph>(sg_config->graph);
    auto src_graph = std::make_shared<Graph>(sg_config->graph);

    ref_graph->initialize(ref_instances);
    src_graph->initialize(src_instances);
    ref_graph->construct_edges();
    src_graph->construct_edges();
    ref_graph->construct_triplets();
    src_graph->construct_triplets();

    // Loop detection
    torch::Tensor ref_node_features, src_node_features;
    std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
    std::vector<float> match_scores;
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids;

    auto loop_detector = std::make_shared<SgNet>(loop_config, weights_folder);
    loop_detector->graph_encoder(ref_graph->get_const_nodes(), ref_node_features);
    loop_detector->graph_encoder(src_graph->get_const_nodes(), src_node_features);
    TORCH_CHECK(src_node_features.device().is_cuda(), "src node feats must be a CUDA tensor")

    loop_detector->detect_loop(src_node_features, ref_node_features, match_pairs, match_scores);

    // Export
    std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> match_instances;
    IO::extract_match_instances(match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_instances);
    IO::extract_instance_correspondences(src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_pairs, match_scores, src_centroids, ref_centroids);
    
    if(visualization>0){
        ROS_WARN("run visualization");
        Visualization::render_point_cloud(ref_map->export_global_pcd(), mypubs.ref_graph,frame_id);
        Visualization::render_point_cloud(src_map->export_global_pcd(), mypubs.src_graph, "local");
        Visualization::instance_centroids(ref_graph->get_centroids(),mypubs.ref_centroids,frame_id,{255,0,0});

        Visualization::instance_match(src_centroids, ref_centroids, mypubs.instance_match);
    }

    ros::spinOnce();
    return 0;

}