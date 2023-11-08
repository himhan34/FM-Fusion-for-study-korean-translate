#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"

#include "Utility.h"
#include "SceneGraph.h"

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
    std::string root_dir =
            utility::GetProgramOptionAsString(argc, argv, "--root");
    std::string active_sequence = 
        utility::GetProgramOptionAsString(argc, argv, "--prev_sequence");


    // init
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
    if(sg_config==nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }
    fmfusion::SceneGraph scene_graph(*sg_config);


    // load
    scene_graph.load(root_dir+"/output/"+active_sequence);

    // Visualize 
    auto &gui_app = gui::Application::GetInstance();  
    gui_app.Initialize(argc, (const char**)argv);  
    gui_app.AddWindow(std::make_shared<InstanceWindow>("Instance Window"));
    gui_app.Run();

    return 0;

}