#include <sstream>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"

#include "open3d_conversions/open3d_conversions.h"

#include "mapping/SemanticMapping.h"
#include "tools/Tools.h"
#include "tools/IO.h"
#include "tools/Eval.h"
#include "sgloop/Graph.h"
#include "sgloop/SGNet.h"
#include "sgloop/ShapeEncoder.h"
#include "sgloop/LoopDetector.h"

#include "registration/Prune.h"

#include "Visualization.h"

struct Publishers{
    ros::Publisher ref_graph, src_graph;
    ros::Publisher ref_centroids, src_centroids;
    ros::Publisher ref_edges, src_edges;
    ros::Publisher instance_match;
}mypubs;

struct GraphMsg{
    sensor_msgs::PointCloud2 global_cloud;
}ref_msgs, src_msgs;

Visualization::VizParam viz_param;

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
    std::string gt_file = nh_private.param("gt_file", std::string(""));
    int visualization = nh_private.param("visualization", 0);
    bool prune_instances = nh_private.param("prune_instances", true);

    viz_param.edge_width = n.param("viz/edge_width", 0.02);
    viz_param.edge_color[0] = n.param("viz/edge_color/r", 0.0);
    viz_param.edge_color[1] = n.param("viz/edge_color/g", 0.0);
    viz_param.edge_color[2] = n.param("viz/edge_color/b", 1.0);
    viz_param.centroid_size = n.param("viz/centroid_size", 0.1);

    // Publisher
    mypubs.ref_graph = nh_private.advertise<sensor_msgs::PointCloud2>("ref_graph", 1000);
    mypubs.src_graph = nh_private.advertise<sensor_msgs::PointCloud2>("src_graph", 1000);
    mypubs.ref_centroids = nh_private.advertise<visualization_msgs::Marker>("ref_centroids", 1000);
    mypubs.src_centroids = nh_private.advertise<visualization_msgs::Marker>("src_centroids", 1000);
    mypubs.ref_edges = nh_private.advertise<visualization_msgs::Marker>("ref_edges", 1000);
    mypubs.src_edges = nh_private.advertise<visualization_msgs::Marker>("src_edges", 1000);
    mypubs.instance_match = nh_private.advertise<visualization_msgs::Marker>("instance_match", 1000);

    assert(set_cfg && set_wegith_folder);
    assert(set_ref_scene && set_src_scene);

    std::cout<<"config file: "<<config_file<<std::endl;

    // Load Config
    Config *sg_config = utility::create_scene_graph_config(config_file, true);
    if (sg_config==nullptr){
        ROS_WARN("Failed to create scene graph config.");
        return 0;
    }

    // Load reconstrcuted maps
    auto ref_mapping = std::make_shared<SemanticMapping>(SemanticMapping(sg_config->mapping_cfg,sg_config->instance_cfg));
    auto src_mapping = std::make_shared<SemanticMapping>(SemanticMapping(sg_config->mapping_cfg,sg_config->instance_cfg));
    ref_mapping->load(ref_scene_dir);
    src_mapping->load(src_scene_dir);
    ref_mapping->extract_bounding_boxes();
    src_mapping->extract_bounding_boxes();

    std::vector<InstanceId> ref_names, src_names;
    std::vector<InstancePtr> ref_instances, src_instances;
    ref_mapping->export_instances(ref_names,ref_instances);
    src_mapping->export_instances(src_names,src_instances);

    // Construct explicit graph
    auto ref_graph = std::make_shared<Graph>(sg_config->graph);
    auto src_graph = std::make_shared<Graph>(sg_config->graph);

    ref_graph->initialize(ref_instances);
    src_graph->initialize(src_instances);
    ref_graph->construct_edges();
    src_graph->construct_edges();
    ref_graph->construct_triplets();
    src_graph->construct_triplets();
    fmfusion::DataDict ref_data_dict = ref_graph->extract_data_dict();
    fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();

    // Encode using SgNet
    auto loop_detector = std::make_shared<fmfusion::LoopDetector>(fmfusion::LoopDetector(
                        sg_config->shape_encoder, sg_config->sgnet, weights_folder));
    fmfusion::o3d_utility::Timer timer;
    timer.Start();                        
    loop_detector->encode_ref_scene_graph(ref_graph->get_const_nodes(), ref_data_dict);
    timer.Stop();
    std::cout<<"Encode ref graph takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";
    loop_detector->encode_src_scene_graph(src_graph->get_const_nodes(), src_data_dict);

    // Hierachical matching
    std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
    std::vector<float> match_scores;
    std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;
    std::vector<float> corr_scores_vec;

    int M; // number of matched nodes
    int C=0; // number of matched points
    std::vector <NodePair> pruned_match_pairs;
    std::vector<float> pruned_match_scores;

    M = loop_detector->match_nodes(match_pairs, match_scores);
    std::vector<bool> pruned_true_masks(M, false);
    ROS_WARN("Matched nodes: %d", M);
    if(prune_instances && M>0){        
        Registration::pruneInsOutliers(sg_config->reg,
                                        src_graph->get_const_nodes(),
                                        ref_graph->get_const_nodes(),
                                        match_pairs,
                                        pruned_true_masks);
    }
    else{
        pruned_true_masks = std::vector<bool>(M, true);
    }

    pruned_match_pairs = fmfusion::utility::update_masked_vec(match_pairs, pruned_true_masks);
    pruned_match_scores = fmfusion::utility::update_masked_vec(match_scores, pruned_true_masks);
    ROS_WARN("Keep %d matched nodes after prune", pruned_match_pairs.size());


    if(pruned_match_pairs.size()>0){ // dense match       
        C = loop_detector->match_instance_points(pruned_match_pairs, corr_src_points, corr_ref_points, corr_scores_vec);
        ROS_WARN("Matched points: %d", C);
    }

    // Export
    std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> pred_instances;
    std::vector<bool> pred_masks;
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
    fmfusion::O3d_Cloud_Ptr src_cloud_ptr, ref_cloud_ptr;
    Eigen::Matrix4d pred_pose;    
    fmfusion::IO::extract_match_instances(
        pruned_match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), pred_instances);
    fmfusion::IO::extract_instance_correspondences(
        src_graph->get_const_nodes(), ref_graph->get_const_nodes(), pruned_match_pairs, pruned_match_scores, src_centroids, ref_centroids);

    if(gt_file.size()>0){
        int count_true = fmfusion::maks_true_instance(gt_file, pred_instances, pred_masks);
        std::cout<<"True instance match: "<<count_true<<"/"<<pred_instances.size()<<std::endl;
    }

    if(visualization>0){
        ROS_WARN("run visualization");
        Visualization::render_point_cloud(ref_mapping->export_global_pcd(), mypubs.ref_graph,frame_id);
        ros::Duration(0.5).sleep();
        Visualization::render_point_cloud(src_mapping->export_global_pcd(), mypubs.src_graph, "local");
        Visualization::instance_centroids(ref_graph->get_centroids(),mypubs.ref_centroids,frame_id,viz_param.centroid_size,viz_param.centroid_color);
        Visualization::instance_centroids(src_graph->get_centroids(),mypubs.src_centroids,"local",viz_param.centroid_size,viz_param.centroid_color);
        Visualization::inter_graph_edges(ref_graph->get_centroids(), ref_graph->get_edges(), mypubs.ref_edges, viz_param.edge_width, viz_param.edge_color , frame_id);
        Visualization::instance_match(src_centroids, ref_centroids, mypubs.instance_match,"world", pred_masks);
    }

    ros::spinOnce();
    return 0;

}