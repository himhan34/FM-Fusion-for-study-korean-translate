// #include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <vector>

#include "open3d/Open3D.h"

#include "Utility.h"
#include "SceneGraph.h"
#include "Common.h"
#include "tools/Tools.h"
#include "sgloop/Graph.h"
// #include "sgloop/LoopDetector.h"

struct InstanceMap{
  std::vector<fmfusion::InstanceId> names;
  std::vector<fmfusion::InstancePtr> instances;
};

int main(int argc, char* argv[]) {

  using namespace open3d;

  std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
  std::string ref_map_dir = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
  std::string src_map_dir = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
  std::string sgnet_dir = utility::GetProgramOptionAsString(argc, argv, "--pth_dir");
  std::string bert_dir = utility::GetProgramOptionAsString(argc, argv, "--bert_dir");
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
  InstanceMap ref_instances, src_instances;
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
  // fmfusion::LoopDetectorConfig loop_config;
  // loop_config.encoder_path = sgnet_dir;
  // loop_config.bert_path = bert_dir;
  // auto loop_detector = std::make_shared<fmfusion::LoopDetector>(fmfusion::LoopDetector(loop_config));

  // torch::Tensor ref_node_features;
  // loop_detector->encode({0}, ref_instances, ref_node_features);


  // visualization
  if (visualize){
    auto ref_geometries = ref_map->get_geometries(true, false);
    auto ref_edge_lineset = fmfusion::visualization::draw_edges(ref_graph->get_const_nodes(), ref_graph->get_const_edges());
    std::vector<fmfusion::O3d_Geometry_Ptr> viz_geometries;
    viz_geometries.insert(viz_geometries.end(), ref_geometries.begin(), ref_geometries.end());
    viz_geometries.emplace_back(ref_edge_lineset);
    open3d::visualization::DrawGeometries(viz_geometries, "UST_RI", 1920, 1080);
  }

  return 0;
}

