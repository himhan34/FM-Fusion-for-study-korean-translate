#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <vector>

#include "open3d/Open3D.h"

#include "Utility.h"
#include "SceneGraph.h"
#include "Common.h"
#include "sgloop/LoopDetector.h"

int main(int argc, char* argv[]) {

  using namespace open3d;

  std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
  std::string ref_map = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
  std::string src_map = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
  std::string sgnet_dir = utility::GetProgramOptionAsString(argc, argv, "--pth_dir");
  std::string bert_dir = utility::GetProgramOptionAsString(argc, argv, "--bert_dir");

  // init
  auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
  if(sg_config==nullptr) {
      utility::LogWarning("Failed to create scene graph config.");
      return 0;
  }

  // Load SG
  auto ref_graph = std::make_shared<fmfusion::SceneGraph>(fmfusion::SceneGraph(*sg_config));
  auto src_graph = std::make_shared<fmfusion::SceneGraph>(fmfusion::SceneGraph(*sg_config));

  ref_graph->load(ref_map);
  src_graph->load(src_map);
  ref_graph->extract_bounding_boxes();
  src_graph->extract_bounding_boxes();

  //
  std::vector<fmfusion::EdgePtr> ref_edges, src_edges;
  fmfusion::construct_edges(ref_graph->export_instances(), ref_edges);
  fmfusion::construct_edges(src_graph->export_instances(), src_edges);

  // Detect loop
  fmfusion::LoopDetectorConfig loop_config;
  loop_config.encoder_path = sgnet_dir;
  loop_config.bert_path = bert_dir;
  auto loop_detector = std::make_shared<fmfusion::LoopDetector>(fmfusion::LoopDetector(loop_config));

  std::vector<fmfusion::InstancePtr> ref_instances = ref_graph->export_instances();
  torch::Tensor ref_node_features;
  loop_detector->encode({0}, ref_instances, ref_node_features);

  return 0;
}

