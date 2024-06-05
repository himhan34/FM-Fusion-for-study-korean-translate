#include <iostream>
#include <memory>
#include <vector>
#include "open3d/Open3D.h"

#include "Utility.h"
#include "SceneGraph.h"
#include "Common.h"
#include "tools/Tools.h"
#include "sgloop/Graph.h"
#include "sgloop/ShapeEncoder.h"
#include "sgloop/SGNet.h"
#include "back_end/reglib.h"

struct ExplicitInstances{
  std::vector<fmfusion::InstanceId> names;
  std::vector<fmfusion::InstancePtr> instances;
  // std::vector<Eigen::Vector3d> xyz;
  // std::vector<uint32_t> labels;
};


Eigen::MatrixX3d vectorToMatrix(const std::vector<Eigen::Vector3d> &vector) {
    // 创建一个动态大小的矩阵，行数为 vector 的大小，每行3列
    Eigen::MatrixX3d matrix(vector.size(), 3);

    // 将 vector 中的每个点复制到矩阵的对应行中
    for (size_t i = 0; i < vector.size(); ++i) {
        matrix.row(i) = vector[i];
    }

    return matrix;
}


void estimate_pose(const std::vector<fmfusion::NodePtr> &src_nodes,
                    const std::vector<fmfusion::NodePtr> &ref_nodes,
                    const std::vector<std::pair<uint32_t, uint32_t>> &match_pairs,
                    const std::vector<float> &match_scores,
                    const std::vector<Eigen::Vector3d> &corr_src_points,
                    const std::vector<Eigen::Vector3d> &corr_ref_points,
                    fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                    fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr,
                    Eigen::Matrix4d &pose) 
{
    std::vector<Eigen::Vector3d> src_centroids;
    std::vector<Eigen::Vector3d> ref_centroids;
    std::stringstream msg;
    msg << match_pairs.size() << " Matched pairs: \n";

    Eigen::MatrixX3d src_corrp(match_pairs.size(), 3);
    Eigen::MatrixX3d ref_corrp(match_pairs.size(), 3);
    std::vector<Eigen::Vector3d> src_points, ref_points;
    int64_t index = 0;
    for (auto pair: match_pairs) {
        auto src_node = src_nodes[pair.first];
        auto ref_node = ref_nodes[pair.second];
        src_centroids.push_back(src_node->centroid);
        src_corrp.row(index) = src_node->centroid;
        src_points.insert(src_points.end(), src_node->cloud->points_.begin(), src_node->cloud->points_.end());
        ref_centroids.push_back(ref_node->centroid);
        ref_corrp.row(index) = ref_node->centroid;
        ref_points.insert(ref_points.end(), ref_node->cloud->points_.begin(), ref_node->cloud->points_.end());
        index++;
        msg << "(" << pair.first << "," << pair.second << ") "
            << "(" << src_node->semantic << "," << ref_node->semantic << ")\n";
    }

    auto src_cloud = vectorToMatrix(src_points);
    auto ref_cloud = vectorToMatrix(ref_points);
    g3reg::Config config;
    config.vertex_info.noise_bound_vec = {0.2, 0.3};
    config.tf_solver = "quatro";
    FRGresult result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud, ref_cloud, config);
    std::cout << "FRG result: \n" << result.tf << "\n";
    std::cout << msg.str() << std::endl;

    src_cloud_ptr = std::make_shared<fmfusion::O3d_Cloud>(src_points);
    ref_cloud_ptr = std::make_shared<fmfusion::O3d_Cloud>(ref_points);
    pose = result.tf;
};

bool save_match_results(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                      const std::vector<float> &match_scores,
                      const std::string &output_file_dir)
{
  std::ofstream output_file(output_file_dir);
  if(!output_file.is_open()){
    std::cerr<<"Failed to open file: "<<output_file_dir<<std::endl;
    return false;
  }

  output_file<<"# src_id,ref_id,score"<<std::endl;
  for (size_t i = 0; i < match_scores.size(); i++)
  {
    output_file<<match_pairs[i].first<<","<<match_pairs[i].second<<","
      <<std::fixed<<std::setprecision(3)<<match_scores[i]<<std::endl;
  }
  output_file.close();
 
  return true;

}

int main(int argc, char* argv[]) 
{
    using namespace open3d;

    std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string ref_map_dir = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    std::string src_map_dir = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string weights_folder = utility::GetProgramOptionAsString(argc, argv, "--weights_folder");
    std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
    bool fused = utility::ProgramOptionExists(argc, argv, "--fusion");
    bool dense_match = utility::ProgramOptionExists(argc, argv, "--dense_match");
    int viz_mode = utility::GetProgramOptionAsInt(argc, argv, "--viz_mode", 0);

    // init
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
    if(sg_config==nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }
    fmfusion::ShapeEncoderConfig shape_config;

    // Load Map
    auto ref_map = std::make_shared<fmfusion::SceneGraph>(fmfusion::SceneGraph(*sg_config));
    auto src_map = std::make_shared<fmfusion::SceneGraph>(fmfusion::SceneGraph(*sg_config));

    ref_map->load(ref_map_dir);
    src_map->load(src_map_dir);
    ref_map->extract_bounding_boxes();
    src_map->extract_bounding_boxes();

    // Export instances
    ExplicitInstances ref_instances, src_instances;
    ref_map->export_instances(ref_instances.names, ref_instances.instances);
    src_map->export_instances(src_instances.names, src_instances.instances);
    
    // Construct GNN from the exported instances
    auto ref_graph  = std::make_shared<fmfusion::Graph>(sg_config->graph);
    auto src_graph  = std::make_shared<fmfusion::Graph>(sg_config->graph);

    ref_graph->initialize(ref_instances.instances);
    src_graph->initialize(src_instances.instances);
    ref_graph->construct_edges();
    src_graph->construct_edges();
    ref_graph->construct_triplets();
    src_graph->construct_triplets();
    fmfusion::DataDict ref_data_dict = ref_graph->extract_data_dict();
    fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();


    // Encode using SgNet
    fmfusion::o3d_utility::Timer timer;

    fmfusion::SgNetConfig sgnet_config;
    auto sgnet = std::make_shared<fmfusion::SgNet>(fmfusion::SgNet(sgnet_config, weights_folder));
    fmfusion::ShapeEncoderPtr shape_encoder = std::make_shared<fmfusion::ShapeEncoder>(shape_config, weights_folder);

    torch::Tensor ref_node_features, src_node_features;
    torch::Tensor ref_node_shape, src_node_shape;
    torch::Tensor ref_node_knn_points, src_node_knn_points; // (Nr,K,3)
    torch::Tensor ref_node_knn_feats, src_node_knn_feats;

    timer.Start();
    shape_encoder->encode(
        ref_data_dict.xyz, ref_data_dict.labels, ref_data_dict.centroids, ref_data_dict.nodes, ref_node_shape, ref_node_knn_points, ref_node_knn_feats);
    timer.Stop();
    std::cout<<"Encode ref point cloud takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";    
    shape_encoder->encode(
        src_data_dict.xyz, src_data_dict.labels, src_data_dict.centroids, src_data_dict.nodes, src_node_shape, src_node_knn_points, src_node_knn_feats);
    sgnet->graph_encoder(ref_graph->get_const_nodes(), ref_node_features);
    sgnet->graph_encoder(src_graph->get_const_nodes(), src_node_features);
    TORCH_CHECK(src_node_features.device().is_cuda(), "src node feats must be a CUDA tensor")
    
    // Fusion
    if(fused){
        src_node_features = torch::cat({src_node_features, src_node_shape}, 1);
        ref_node_features = torch::cat({ref_node_features, ref_node_shape}, 1);
    }

    // Hierachical matching
    std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
    std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> match_instances;
    std::vector<float> match_scores;
    std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;

    int M; // number of matched nodes
    int C=0; // number of matched points

    sgnet->match_nodes(src_node_features, ref_node_features, match_pairs, match_scores,fused);
    M = match_pairs.size();
    if(dense_match && M>0){
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

    // Estimate pose
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
    fmfusion::O3d_Cloud_Ptr src_cloud_ptr, ref_cloud_ptr;
    Eigen::Matrix4d pred_pose;    
    fmfusion::IO::extract_match_instances(
        match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_instances);
    fmfusion::IO::extract_instance_correspondences(
        src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_pairs, match_scores, src_centroids, ref_centroids);
    estimate_pose(src_graph->get_const_nodes(), 
                ref_graph->get_const_nodes(), 
                match_pairs, 
                match_scores, 
                corr_src_points,
                corr_ref_points,
                src_cloud_ptr, 
                ref_cloud_ptr, 
                pred_pose);

    // visualization
    if (viz_mode==1){ // instance match
        auto ref_geometries = ref_map->get_geometries(true, false);
        // auto ref_edge_lineset = fmfusion::visualization::draw_edges(ref_graph->get_const_nodes(), ref_graph->get_const_edges());
        auto instance_match_lineset = fmfusion::visualization::draw_instance_correspondences(src_centroids, ref_centroids);

        std::vector<fmfusion::O3d_Geometry_Ptr> viz_geometries;
        viz_geometries.insert(viz_geometries.end(), ref_geometries.begin(), ref_geometries.end());
        // viz_geometries.emplace_back(ref_edge_lineset);
        viz_geometries.emplace_back(instance_match_lineset);

        open3d::visualization::DrawGeometries(viz_geometries, "UST_RI", 1920, 1080);
    }
    else if (viz_mode==2){   // registration result
        src_cloud_ptr->Transform(pred_pose);
        std::vector<fmfusion::O3d_Geometry_Ptr> viz_geometries = {src_cloud_ptr, ref_cloud_ptr};
        open3d::visualization::DrawGeometries(viz_geometries, "UST_RI", 1920, 1080);
    }
    
    if(output_folder.size()>0){
        // save the instance correspondences
        std::string ref_scene = ref_map_dir.substr(ref_map_dir.find_last_of("/")+1);
        std::string src_scene = src_map_dir.substr(src_map_dir.find_last_of("/")+1);
        std::string pair_name = src_scene+"-"+ref_scene;
        std::string output_file_dir = output_folder+"/"+pair_name+".txt";
        // save_match_results(match_instances, match_scores, output_file_dir);
        fmfusion::IO::save_match_results(pred_pose,match_instances, match_scores, output_file_dir);
        open3d::io::WritePointCloudToPLY(output_folder+"/"+src_scene+".ply", *src_cloud_ptr, {}); // open3d::io::WritePointCloudOption);
        if(C>0){
            fmfusion::O3d_Cloud_Ptr corr_src_pcd = std::make_shared<fmfusion::O3d_Cloud>(corr_src_points);
            fmfusion::O3d_Cloud_Ptr corr_ref_pcd = std::make_shared<fmfusion::O3d_Cloud>(corr_ref_points);
            open3d::io::WritePointCloudToPLY(output_folder+"/"+pair_name+"_csrc.ply", *corr_src_pcd, {});
            open3d::io::WritePointCloudToPLY(output_folder+"/"+pair_name+"_cref.ply", *corr_ref_pcd, {});
        }

        std::cout<<"Save output result to "<<output_folder<<std::endl;
    }

    return 0;
}

