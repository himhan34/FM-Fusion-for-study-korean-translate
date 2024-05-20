#include <iostream>
#include <memory>
#include <vector>
#include "open3d/Open3D.h"

#include "Utility.h"
#include "SceneGraph.h"
#include "Common.h"
#include "tools/Tools.h"
#include "sgloop/Graph.h"
#include "sgloop/LoopDetector.h"
#include "back_end/reglib.h"

struct ExplicitInstances {
    std::vector<fmfusion::InstanceId> names;
    std::vector<fmfusion::InstancePtr> instances;
};

void extract_match_instances(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            const std::vector<fmfusion::NodePtr> &src_nodes,
                            const std::vector<fmfusion::NodePtr> &ref_nodes,
                            std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> &match_instances)
{
  for (auto pair: match_pairs){
    auto src_node = src_nodes[pair.first];
    auto ref_node = ref_nodes[pair.second];
    match_instances.push_back(std::make_pair(src_node->instance_id, ref_node->instance_id));
  }

}

void extract_instance_correspondences(const std::vector<fmfusion::NodePtr> &src_nodes, const std::vector<fmfusion::NodePtr> &ref_nodes, 
  const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, const std::vector<float> &match_scores,
  std::vector<Eigen::Vector3d> &src_centroids, std::vector<Eigen::Vector3d> &ref_centroids)
{
  // std::vector<Eigen::Vector3d> src_centroids;
  // std::vector<Eigen::Vector3d> ref_centroids;
  std::stringstream msg;
  msg<<match_pairs.size()<<"Matched pairs: \n";

  for (auto pair: match_pairs){
    auto src_node = src_nodes[pair.first];
    auto ref_node = ref_nodes[pair.second];
    src_centroids.push_back(src_node->centroid);
    ref_centroids.push_back(ref_node->centroid);
    msg<<"("<<src_node->instance_id<<","<<ref_node->instance_id<<") "
      <<"("<<src_node->semantic<<","<<ref_node->semantic<<")\n";
  }

  std::cout<<msg.str()<<std::endl;
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

int main(int argc, char* argv[]) {

  using namespace open3d;

  std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
  std::string ref_map_dir = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
  std::string src_map_dir = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
  std::string weights_folder = utility::GetProgramOptionAsString(argc, argv, "--weights_folder");
  std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
  bool visualize = utility::ProgramOptionExists(argc, argv, "--visualize");

  // init
  auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
  if(sg_config==nullptr) {
      utility::LogWarning("Failed to create scene graph config.");
      return 0;
  }

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
  auto ref_graph  = std::make_shared<fmfusion::Graph>(sg_config->gnn);
  auto src_graph  = std::make_shared<fmfusion::Graph>(sg_config->gnn);
  ref_graph->initialize(ref_instances.instances);
  src_graph->initialize(src_instances.instances);
  ref_graph->construct_edges();
  src_graph->construct_edges();
  ref_graph->construct_triplets();
  src_graph->construct_triplets();

  // Detect loop
  fmfusion::LoopDetectorConfig loop_config;
  auto loop_detector = std::make_shared<fmfusion::LoopDetector>(fmfusion::LoopDetector(loop_config, weights_folder));
  torch::Tensor ref_node_features, src_node_features;
  std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
  std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> match_instances;
  std::vector<float> match_scores;
  std::vector<Eigen::Vector3d> src_centroids, ref_centroids;

  loop_detector->graph_encoder(ref_graph->get_const_nodes(), ref_node_features);
  loop_detector->graph_encoder(src_graph->get_const_nodes(), src_node_features);
  
  loop_detector->detect_loop(src_node_features, ref_node_features, match_pairs, match_scores);
  extract_match_instances(match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_instances);
  extract_instance_correspondences(src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_pairs, match_scores,
                                    src_centroids, ref_centroids);

  // visualization
  if (visualize){
    auto ref_geometries = ref_map->get_geometries(true, false);
    // auto ref_edge_lineset = fmfusion::visualization::draw_edges(ref_graph->get_const_nodes(), ref_graph->get_const_edges());
    auto instance_match_lineset = fmfusion::visualization::draw_instance_correspondences(src_centroids, ref_centroids);

    std::vector<fmfusion::O3d_Geometry_Ptr> viz_geometries;
    viz_geometries.insert(viz_geometries.end(), ref_geometries.begin(), ref_geometries.end());
    // viz_geometries.emplace_back(ref_edge_lineset);
    viz_geometries.emplace_back(instance_match_lineset);

    open3d::visualization::DrawGeometries(viz_geometries, "UST_RI", 1920, 1080);
  }

  if(output_folder.size()>0){
    // save the instance correspondences
    std::string ref_scene = ref_map_dir.substr(ref_map_dir.find_last_of("/")+1);
    std::string src_scene = src_map_dir.substr(src_map_dir.find_last_of("/")+1);
    std::string output_file_dir = output_folder+"/"+src_scene+"-"+ref_scene+".txt";
    save_match_results(match_instances, match_scores, output_file_dir);
    std::cout<<"Save output result to "<<output_file_dir<<std::endl;
  }

  return 0;
}

