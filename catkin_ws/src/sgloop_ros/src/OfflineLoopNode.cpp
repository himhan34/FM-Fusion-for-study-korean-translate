#include <sstream>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"

#include "open3d_conversions/open3d_conversions.h"

#include "SceneGraph.h"
#include "tools/Tools.h"
#include "tools/Eval.h"
#include "sgloop/Graph.h"
#include "sgloop/SGNet.h"
#include "sgloop/ShapeEncoder.h"

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
    std::string gt_file = nh_private.param("gt_file", std::string(""));
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
    Config *sg_config = utility::create_scene_graph_config(config_file, true);
    if (sg_config==nullptr){
        ROS_WARN("Failed to create scene graph config.");
        return 0;
    }
    fmfusion::ShapeEncoderConfig shape_config;

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
    fmfusion::DataDict ref_data_dict = ref_graph->extract_data_dict();
    fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();

    // Encode using SgNet
    SgNetConfig sgnet_config;
    torch::Tensor ref_node_features, src_node_features;
    torch::Tensor ref_node_shape, src_node_shape;
    torch::Tensor ref_node_knn_points, src_node_knn_points; // (Nr,K,3)
    torch::Tensor ref_node_knn_feats, src_node_knn_feats;

    auto sgnet = std::make_shared<SgNet>(sgnet_config, weights_folder);
    fmfusion::ShapeEncoderPtr shape_encoder = std::make_shared<fmfusion::ShapeEncoder>(shape_config, weights_folder);
    fmfusion::o3d_utility::Timer timer;
    timer.Start();
    shape_encoder->encode(
        ref_data_dict.xyz, ref_data_dict.labels, ref_data_dict.centroids, ref_data_dict.nodes, ref_node_shape, ref_node_knn_points, ref_node_knn_feats);
    timer.Stop();
    std::cout<<"Encode ref graph shape takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";
    timer.Start();
    shape_encoder->encode(
        src_data_dict.xyz, src_data_dict.labels, src_data_dict.centroids, src_data_dict.nodes, src_node_shape, src_node_knn_points, src_node_knn_feats);
    std::cout<<"Encode src graph shape takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";

    sgnet->graph_encoder(ref_graph->get_const_nodes(), ref_node_features);
    sgnet->graph_encoder(src_graph->get_const_nodes(), src_node_features);
    TORCH_CHECK(src_node_features.device().is_cuda(), "src node feats must be a CUDA tensor")

    // Hierachical matching
    std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
    std::vector<float> match_scores;
    std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;

    int M; // number of matched nodes
    int C=0; // number of matched points

    sgnet->match_nodes(src_node_features, ref_node_features, match_pairs, match_scores);
    M = match_pairs.size();
    if(M>0){
        torch::Tensor corr_points;
        torch::Tensor corr_scores;        
        float match_pairs_array[2][M]; // 
        for(int i=0;i<M;i++){
            match_pairs_array[0][i] = int(match_pairs[i].first);  // src_node
            match_pairs_array[1][i] = int(match_pairs[i].second); // ref_node
        }
        torch::Tensor src_corr_nodes = torch::from_blob(match_pairs_array[0], {M}).to(torch::kInt64).to(torch::kCUDA);
        torch::Tensor ref_corr_nodes = torch::from_blob(match_pairs_array[1], {M}).to(torch::kInt64).to(torch::kCUDA);

        torch::Tensor src_guided_knn_points = src_node_knn_points.index_select(0, src_corr_nodes);
        torch::Tensor src_guided_knn_feats = src_node_knn_feats.index_select(0, src_corr_nodes);
        torch::Tensor ref_guided_knn_points = ref_node_knn_points.index_select(0, ref_corr_nodes);
        torch::Tensor ref_guided_knn_feats = ref_node_knn_feats.index_select(0, ref_corr_nodes);
        TORCH_CHECK(src_guided_knn_feats.device().is_cuda(), "src guided knn feats must be a CUDA tensor");
        timer.Start();
        C = sgnet->match_points(src_guided_knn_feats, ref_guided_knn_feats, corr_points, corr_scores);
        timer.Stop();
        std::cout<<"Match points takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";
        fmfusion::extract_corr_points(src_guided_knn_points, ref_guided_knn_points, corr_points, corr_src_points, corr_ref_points);
    }


    // Export
    std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> pred_instances;
    std::vector<bool> pred_masks;
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
    fmfusion::O3d_Cloud_Ptr src_cloud_ptr, ref_cloud_ptr;
    Eigen::Matrix4d pred_pose;    
    fmfusion::IO::extract_match_instances(
        match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), pred_instances);
    fmfusion::IO::extract_instance_correspondences(
        src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_pairs, match_scores, src_centroids, ref_centroids);

    if(gt_file.size()>0){
        int count_true = fmfusion::maks_true_instance(gt_file, pred_instances, pred_masks);
        std::cout<<"True instance match: "<<count_true<<"/"<<pred_instances.size()<<std::endl;
    }

    if(visualization>0){
        ROS_WARN("run visualization");
        Visualization::render_point_cloud(ref_map->export_global_pcd(), mypubs.ref_graph,frame_id);
        ros::Duration(0.5).sleep();
        Visualization::render_point_cloud(src_map->export_global_pcd(), mypubs.src_graph, "local");
        Visualization::instance_centroids(ref_graph->get_centroids(),mypubs.ref_centroids,frame_id,{255,0,0});

        Visualization::instance_match(src_centroids, ref_centroids, mypubs.instance_match,"world", pred_masks);
    }

    ros::spinOnce();
    return 0;

}