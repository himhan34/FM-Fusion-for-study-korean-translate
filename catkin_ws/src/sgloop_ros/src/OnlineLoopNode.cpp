#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>

#include "tools/Utility.h"
#include "tools/IO.h"
#include "mapping/SemanticMapping.h"
#include "Visualization.h"

typedef fmfusion::IO::RGBDFrameDirs RGBDFrameDirs;

struct Publishers{
    ros::Publisher rgb_camera;
}mypub;

int main(int argc, char **argv)
{
    using namespace fmfusion;
    using namespace open3d::utility;
    using namespace open3d::io;

    ros::init(argc, argv, "LoopNode");
    ros::NodeHandle n;
    ros::NodeHandle nh_private("~");

    // Settings
    std::string config_file, weights_folder;
    std::string ref_scene_dir, root_dir;

    mypub.rgb_camera = nh_private.advertise<sensor_msgs::Image>("/rgb/image_raw", 1);

    assert(nh_private.getParam("cfg_file", config_file));
    bool set_wegith_folder = nh_private.getParam("weights_folder", weights_folder);
    bool set_ref_scene = nh_private.getParam("ref_scene_dir", ref_scene_dir);
    bool set_src_scene = nh_private.getParam("active_sequence_dir", root_dir);
    int frame_gap = nh_private.param("frame_gap", 1);
    std::string prediction_folder = nh_private.param("prediction_folder", std::string("prediction_no_augment"));
    std::string output_folder = nh_private.param("output_folder", std::string(""));
    std::string frame_id = nh_private.param("frame_id", std::string("world"));
    int visualization = nh_private.param("visualization", 0);

    SetVerbosityLevel((VerbosityLevel)5);
    LogInfo("Read configuration from {:s}",config_file);
    LogInfo("Read RGBD sequence from {:s}", root_dir);
    std::string sequence_name = *filesystem::GetPathComponents(root_dir).rbegin();
    auto global_config = utility::create_scene_graph_config(config_file, true);
    std::cout<<sequence_name<<"\n";

    // Load frames information
    std::vector<RGBDFrameDirs> rgbd_table;
    std::vector<Eigen::Matrix4d> pose_table;
    bool read_ret = fmfusion::IO::construct_preset_frame_table(root_dir,"data_association.txt","trajectory.log",rgbd_table,pose_table);
    if(!read_ret || global_config==nullptr) {
        return 0;
    }

    // Mapping module
    fmfusion::SemanticMapping semantic_mapping(global_config->mapping_cfg, global_config->instance_cfg);
    open3d::geometry::Image depth, color;
    int prev_frame_id = -100;

    // Running sequence
    for(int k=0;k<rgbd_table.size();k++){
        RGBDFrameDirs frame_dirs = rgbd_table[k];
        std::string frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
        frame_name = frame_name.substr(0,frame_name.find_last_of("."));
        int frame_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));

        if((frame_id-prev_frame_id)<frame_gap) continue;
        LogInfo("Processing frame {:s} ...", frame_name);

        ReadImage(frame_dirs.second, depth);
        ReadImage(frame_dirs.first, color);

        auto rgb_cv = std::make_shared<cv::Mat>(color.height_,color.width_,CV_8UC3);
        memcpy(rgb_cv->data,color.data_.data(),color.data_.size()*sizeof(uint8_t));

        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, global_config->mapping_cfg.depth_scale, global_config->mapping_cfg.depth_max, false);
        
        std::vector<fmfusion::DetectionPtr> detections;
        bool loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, frame_name, 
                                                        global_config->mapping_cfg, global_config->instance_cfg.intrinsic.width_, global_config->instance_cfg.intrinsic.height_,
                                                        detections);
        if(loaded){
            semantic_mapping.integrate(frame_id,rgbd, pose_table[k], detections);
            prev_frame_id = frame_id;
        }

        if(true){
            Visualization::render_image(*rgb_cv, mypub.rgb_camera, "world");
        }
    }
    LogWarning("Finished sequence");

    // Post-process
    semantic_mapping.extract_point_cloud();
    semantic_mapping.merge_overlap_instances();
    semantic_mapping.merge_overlap_structural_instances();
    semantic_mapping.extract_bounding_boxes();

    // Save
    LogWarning("Save sequence to {:s}", output_folder+"/"+sequence_name);
    semantic_mapping.Save(output_folder+"/"+sequence_name);

    ros::spin();
    return 0;
}