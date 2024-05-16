#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"

#include "Utility.h"
#include "SceneGraph.h"
#include "Common.h"

using namespace open3d::visualization;


int main(int argc, char *argv[]) 
{
    using namespace open3d;

    std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string map_folder = utility::GetProgramOptionAsString(argc, argv, "--map_folder");
    std::string scene_name = utility::GetProgramOptionAsString(argc, argv, "--scene_name");
    bool visualization = utility::ProgramOptionExists(argc, argv, "--visualization");


    // init
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
    if(sg_config==nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }
    auto global_scene_graph = std::make_shared<fmfusion::SceneGraph>(fmfusion::SceneGraph(*sg_config));

    // Pose Graph
    auto pose_graph_io = std::make_shared<fmfusion::PoseGraphFile>();
    if(open3d::io::ReadIJsonConvertible(map_folder+"/"+scene_name+"/pose_map.json", *pose_graph_io)){
        fmfusion::o3d_utility::LogInfo("Read pose graph from {}.",scene_name);
    }
    auto pose_map = pose_graph_io->poses_;

    //
    int count=0;
    for (auto it=pose_map.begin();it!=pose_map.end();it++){
        fmfusion::o3d_utility::LogInfo("Scene: {}",it->first);
        std::string scene_folder = map_folder+"/"+it->first;
        auto local_sg = std::make_shared<fmfusion::SceneGraph>(fmfusion::SceneGraph(*sg_config));

        local_sg->load(scene_folder);
        local_sg->Transform(it->second);
        std::vector<fmfusion::InstanceId> local_instance_names;
        std::vector<fmfusion::InstancePtr> local_instances;
        local_sg->export_instances(local_instance_names, local_instances);
        global_scene_graph->merge_other_instances(local_instances);

        if(count>0) global_scene_graph->merge_overlap_instances();

        count ++;
        // break;
    }
    global_scene_graph->extract_bounding_boxes();

    //
    if (visualization){
        auto geometries = global_scene_graph->get_geometries(true,true);
        open3d::visualization::DrawGeometries(geometries, scene_name, 1920, 1080);
    }
    
    // Export 
    global_scene_graph->Save(map_folder+"/"+scene_name);

    return 0;

}
