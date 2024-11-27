#include <sstream>
#include <unordered_map>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/PointCloud2.h"
#include "open3d_conversions/open3d_conversions.h"

#include "tools/IO.h"

#include "Visualization.h"
#include "mapping/SemanticMapping.h"

int main(int argc, char **argv)
{
    using namespace open3d::utility::filesystem;
    using namespace fmfusion;

    ros::init(argc, argv, "RenderNode");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    // Params
    std::string config_file;
    std::string scene_result_folder;
    assert(nh_private.getParam("cfg_file", config_file));
    assert(nh_private.getParam("sequence_result_folder", scene_result_folder));
    std::string agent_name = nh_private.param("agent_name", std::string("agentA"));
    std::string map_name = nh_private.param("map_name", std::string(""));
    int iter_num = nh_private.param("iter_num", 3);
    std::string sequence_name = *GetPathComponents(scene_result_folder).rbegin();
    ROS_WARN("Render %s semantic mapping results from %s", agent_name.c_str(),
                                                        scene_result_folder.c_str());

    // Init
    Config *global_config;
    SemanticMapping *semantic_mapping;
    Visualization::Visualizer viz(nh,nh_private);
    {
        global_config = utility::create_scene_graph_config(config_file, false);
        semantic_mapping = new SemanticMapping(global_config->mapping_cfg, global_config->instance_cfg);
    }

    // Load
    semantic_mapping->load(scene_result_folder);
    semantic_mapping->refresh_all_semantic_dict();
    semantic_mapping->merge_floor(false);
    semantic_mapping->merge_overlap_instances();

    // Viz
    for(int i=0;i<iter_num;i++){
        O3d_Cloud_Ptr render_map_pcd;
        if(map_name.size()>0){
            render_map_pcd = open3d::io::CreatePointCloudFromFile(scene_result_folder+"/"+map_name);
            ROS_WARN("Load and render map from %s", map_name.c_str());
        }
        else
            render_map_pcd = semantic_mapping->export_global_pcd(true,0.05);

        Visualization::render_semantic_map(render_map_pcd,
                                        semantic_mapping->export_instance_centroids(0),
                                        semantic_mapping->export_instance_annotations(0),
                                        viz,
                                        agent_name);
        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }


    return 0;

}
