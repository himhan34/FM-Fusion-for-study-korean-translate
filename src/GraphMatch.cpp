#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"

#include "tools/Utility.h"
#include "mapping/SemanticMapping.h"

using namespace open3d::visualization;

class InstanceWindow: public gui::Window{
    // using Super = gui::window;

public:
    InstanceWindow(const std::string &title)
        : gui::Window(title, 1280, 720) {
        // gui::Application::GetInstance().Initialize();
        // gui::Application::GetInstance().AddWindow(this);
        auto pcd = open3d::io::CreatePointCloudFromFile("/media/lch/SeagateExp/dataset_/FusionPortable/output/bday_04/instance_map.ply");
        std::cout<<"pcd size: "<<pcd->points_.size()<<std::endl;
    }


};

int main(int argc, char *argv[]) 
{
    using namespace open3d;

    std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string map_folder =
            utility::GetProgramOptionAsString(argc, argv, "--map_folder");
    std::string src_sequence = 
        utility::GetProgramOptionAsString(argc, argv, "--src_sequence");
    std::string tar_sequence = 
        utility::GetProgramOptionAsString(argc, argv, "--tar_sequence");


    // init
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
    if(sg_config==nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }
    fmfusion::SemanticMapping scene_graph_src(sg_config->mapping_cfg, sg_config->instance_cfg);
    fmfusion::SemanticMapping scene_graph_tar(sg_config->mapping_cfg, sg_config->instance_cfg);

    // load
    scene_graph_src.load(map_folder+"/"+src_sequence);

    // Update 
    scene_graph_src.merge_overlap_instances();
    scene_graph_src.extract_bounding_boxes();
    scene_graph_src.merge_overlap_structural_instances();
    if(tar_sequence!=""){
        fmfusion::o3d_utility::LogInfo("Todo {}.",tar_sequence);
        Eigen::Matrix4d T_tar_offset;
        T_tar_offset<<1,0,0,0,
            0,1,0,0,
            0,0,1,6.0,
            0,0,0,1;
        scene_graph_tar.load(map_folder+"/"+tar_sequence);
        scene_graph_tar.Transform(T_tar_offset);
        scene_graph_tar.merge_overlap_instances();
        scene_graph_tar.extract_bounding_boxes();
        scene_graph_tar.merge_overlap_structural_instances();
    }

    // Visualize
    auto geometries = scene_graph_src.get_geometries(true,true);
    if(!scene_graph_tar.is_empty()){
        auto geometries_tar = scene_graph_tar.get_geometries(true,true);
        geometries.insert(geometries.end(),geometries_tar.begin(),geometries_tar.end());
        std::cout<<"Update tar scene graph from "<<tar_sequence<<std::endl;
    }

    open3d::visualization::DrawGeometries(geometries, src_sequence, 1920, 1080);
    

    return 0;

}