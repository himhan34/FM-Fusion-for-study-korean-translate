#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include "tf/transform_listener.h"

#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>

#include "tools/Utility.h"
#include "tools/IO.h"
#include "tools/TicToc.h"
#include "mapping/SemanticMapping.h"

#include "Visualization.h"


int main(int argc, char **argv)
{
    using namespace fmfusion;
    using namespace open3d::utility;
    using namespace open3d::io;

    ros::init(argc, argv, "MappingNode");
    ros::NodeHandle n;
    ros::NodeHandle nh_private("~");

    // Settings
    std::string config_file;
    std::string root_dir;
    std::string LOCAL_AGENT;

    assert(nh_private.getParam("cfg_file", config_file));
    assert(nh_private.getParam("active_sequence_dir", root_dir));
    nh_private.getParam("local_agent", LOCAL_AGENT);
    int frame_gap = nh_private.param("frame_gap", 1);
    std::string prediction_folder = nh_private.param("prediction_folder", std::string("prediction_no_augment"));
    std::string output_folder = nh_private.param("output_folder", std::string(""));
    std::string association_name = nh_private.param("association_name", std::string(""));
    std::string trajectory_name = nh_private.param("trajectory_name", std::string("trajectory.log"));
    int o3d_verbose_level = nh_private.param("o3d_verbose_level", 2);
    int visualization = nh_private.param("visualization", 0);
    ROS_WARN("MappingNode started");

    // Inits
    Config *global_config;
    SemanticMapping *semantic_mapping;
    std::string sequence_name = *filesystem::GetPathComponents(root_dir).rbegin();
    Visualization::Visualizer viz(n,nh_private);
    {
        global_config = utility::create_scene_graph_config(config_file, true);
        SetVerbosityLevel((VerbosityLevel)o3d_verbose_level);
        if(output_folder.size()>0 && !filesystem::DirectoryExists(output_folder)) 
            filesystem::MakeDirectory(output_folder);

        std::ofstream out_file(output_folder+"/config.txt");
        out_file<<utility::config_to_message(*global_config);
        out_file.close();

        semantic_mapping = new SemanticMapping(global_config->mapping_cfg, global_config->instance_cfg);
    }

    // Load frames information
    std::vector<fmfusion::IO::RGBDFrameDirs> rgbd_table;
    std::vector<Eigen::Matrix4d> pose_table;
    if(association_name.empty()) {
        ROS_WARN("--- Read all RGB-D frames in %s ---",root_dir.c_str());
        IO::construct_sorted_frame_table(root_dir,
                                        rgbd_table,
                                        pose_table);
        if(rgbd_table.size()>4000) return 0;
    }
    else{
        bool read_ret = IO::construct_preset_frame_table(root_dir,
                                                        association_name,
                                                        trajectory_name,
                                                        rgbd_table,
                                                        pose_table);
        if(!read_ret) return 0;
    }


    open3d::geometry::Image depth, color;
    int prev_frame_id = -100;
    int prev_save_frame = -100;
    fmfusion::TicTocSequence tic_toc_seq("# Load Integration Export", 3);

    for(int k=0;k<rgbd_table.size();k++){
        IO::RGBDFrameDirs frame_dirs = rgbd_table[k];
        std::string frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
        frame_name = frame_name.substr(0,frame_name.find_last_of("."));
        int seq_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));
        if((seq_id-prev_frame_id)<frame_gap) continue;
        // map_timing.create_frame(seq_id);
        std::cout<<"Processing frame "<<frame_name<<" ..."<<std::endl;

        //
        bool loaded;
        std::vector<DetectionPtr> detections;
        std::shared_ptr<open3d::geometry::RGBDImage> rgbd;
        { // load RGB-D, detections
            ReadImage(frame_dirs.second, depth);
            ReadImage(frame_dirs.first, color);
            rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(
                color, depth, global_config->mapping_cfg.depth_scale, global_config->mapping_cfg.depth_max, false);
            
            loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, frame_name, 
                                                            global_config->mapping_cfg, global_config->instance_cfg.intrinsic.width_, global_config->instance_cfg.intrinsic.height_,
                                                            detections);
        }
        if(!loaded) continue;

        semantic_mapping->integrate(seq_id,rgbd, pose_table[k], detections);
        prev_frame_id = seq_id;

        { // Viz poses
            Visualization::render_camera_pose(pose_table[k], viz.camera_pose, LOCAL_AGENT, seq_id);
            Visualization::render_path(pose_table[k], viz.path_msg, viz.path, LOCAL_AGENT, seq_id);
        }

        if(viz.pred_image.getNumSubscribers()>0){ // Viz images
            std::string pred_img_dir = root_dir+"/pred_viz/"+frame_name+".jpg";
            cv::Mat pred_img = cv::imread(pred_img_dir);
            Visualization::render_image(pred_img, viz.pred_image, LOCAL_AGENT);
        }

        {// Viz 3D
            std::vector<InstanceId> valid_names;
            std::vector<InstancePtr> valid_instances;
            O3d_Cloud_Ptr global_instance_pcd;
            std::vector<Eigen::Vector3d> src_instance_centroids;
            std::vector<std::string> src_instance_annotations;

            src_instance_centroids = semantic_mapping->export_instance_centroids(0);
            src_instance_annotations = semantic_mapping->export_instance_annotations(0);
            semantic_mapping->export_instances(valid_names, valid_instances, 0);
            global_instance_pcd = semantic_mapping->export_global_pcd(true,0.05);

            Visualization::instance_centroids(src_instance_centroids,
                                            viz.src_centroids,
                                            LOCAL_AGENT,
                                            viz.param.centroid_size,
                                            viz.param.centroid_color);
            Visualization::node_annotation(src_instance_centroids,
                                            src_instance_annotations,
                                            viz.node_annotation,
                                            LOCAL_AGENT,
                                            viz.param.annotation_size,
                                            viz.param.annotation_voffset,
                                            viz.param.annotation_color);

            Visualization::render_point_cloud(global_instance_pcd, viz.src_graph, LOCAL_AGENT); 

            ROS_INFO("Render %d instances and %d points",valid_names.size(),
                                                        global_instance_pcd->points_.size());

        }
    
    }

    ROS_WARN("Finished sequence with %d frames",rgbd_table.size());

    // Pose-process
    semantic_mapping->extract_point_cloud();
    semantic_mapping->merge_floor();
    semantic_mapping->merge_overlap_instances();
    semantic_mapping->merge_overlap_structural_instances();

    // Save
    semantic_mapping->Save(output_folder+"/"+sequence_name);
    tic_toc_seq.export_data(output_folder+"/"+sequence_name+"/time_records.txt");
    fmfusion::utility::write_config(output_folder+"/"+sequence_name+"/config.txt",*global_config);


    return 0;
}
