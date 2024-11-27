#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "tools/Utility.h"
#include "tools/IO.h"
#include "mapping/SemanticMapping.h"
#include "tools/TicToc.h"

typedef fmfusion::IO::RGBDFrameDirs RGBDFrameDirs;

int main(int argc, char *argv[]) 
{
    using namespace open3d;

    std::string config_file = 
            utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string root_dir =
            utility::GetProgramOptionAsString(argc, argv, "--root");
    std::string association_name = 
            utility::GetProgramOptionAsString(argc, argv, "--association",""); //data_association.txt
    std::string trajectory_name = 
            utility::GetProgramOptionAsString(argc, argv, "--trajectory","trajectory.log");
    std::string prediction_folder = 
            utility::GetProgramOptionAsString(argc, argv, "--prediction","prediction_no_augment");
    std::string output_folder = 
            utility::GetProgramOptionAsString(argc, argv, "--output","./");
    std::string output_subseq = 
            utility::GetProgramOptionAsString(argc, argv, "--subseq","");
    int max_frames =
            utility::GetProgramOptionAsInt(argc, argv, "--max_frames", 5000);
    int verbose = 
            utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    bool visualize_flag = 
            utility::ProgramOptionExists(argc, argv, "--visualization");
    bool global_tsdf = 
            utility::ProgramOptionExists(argc,argv,"--global_tsdf");
    int frame_gap = 
            utility::GetProgramOptionAsInt(argc, argv, "--frame_gap", 2); // semantic mapping in every frame_gap frames
    int save_instances_gap = 
            utility::GetProgramOptionAsInt(argc, argv, "--save_instance_gap", 50000);
    int save_global_map_gap = 
            utility::GetProgramOptionAsInt(argc, argv, "--save_global_map_gap", -1);
    // Save the frame-wise instance centroid and semantic label
    int save_frame_instances = 
            utility::GetProgramOptionAsInt(argc, argv, "--save_frame_instances", -1);

    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    utility::LogInfo("Read configuration from {:s}",config_file);
    utility::LogInfo("Read RGBD sequence from {:s}", root_dir);
    std::string sequence_name = *utility::filesystem::GetPathComponents(root_dir).rbegin();
    std::string scene_output = output_folder+"/"+sequence_name+output_subseq;
    if(!utility::filesystem::DirectoryExists(scene_output)) utility::filesystem::MakeDirectory(scene_output);

    if (save_frame_instances>0){
        if(!utility::filesystem::DirectoryExists(root_dir+"/hydra_lcd")) 
            utility::filesystem::MakeDirectory(root_dir+"/hydra_lcd");
    }

    // Load frames information
    std::vector<RGBDFrameDirs> rgbd_table;
    std::vector<Eigen::Matrix4d> pose_table;
    if(association_name.empty()) {
        utility::LogInfo("--- Read all RGB-D frames in {:s} ---",root_dir);
        fmfusion::IO::construct_sorted_frame_table(root_dir,rgbd_table,pose_table);
        // utility::LogInfo("find {:d} rgb-d frame with pose",rgbd_table.size());
        if(rgbd_table.size()>4000) return 0;
    }
    else{
        bool read_ret = fmfusion::IO::construct_preset_frame_table(root_dir,association_name,trajectory_name,rgbd_table,pose_table);
        if(!read_ret) return 0;
    }

    if(rgbd_table.empty()) {
        utility::LogWarning("No RGB-D frames found in {:s}",root_dir);
        return 0;
    }

    visualization::Visualizer vis;

    // if (visualize_flag) vis.CreateVisualizerWindow("Open3D", 1600, 900);
    
    //
    auto global_config = fmfusion::utility::create_scene_graph_config(config_file, true);

    if(global_config==nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }

    //
    switch (global_config->dataset)
    {
    case fmfusion::Config::DATASET_TYPE::SCANNET:
        utility::LogInfo("Dataset: scannet");
        break;
    case fmfusion::Config::DATASET_TYPE::FUSION_PORTABLE:
        utility::LogInfo("Dataset: fusion portable");
        break;
    case fmfusion::Config::DATASET_TYPE::REALSENSE:
        utility::LogInfo("Dataset: Realsense");
        break;
    case fmfusion::Config::DATASET_TYPE::MATTERPORT:
        utility::LogInfo("Dataset: matterport");
        break;
    case fmfusion::Config::DATASET_TYPE::RIO:
        utility::LogInfo("Dataset: RIO");
        break;
    }

    fmfusion::SemanticMapping semantic_mapping(global_config->mapping_cfg, global_config->instance_cfg);

    pipelines::integration::ScalableTSDFVolume global_volume(global_config->instance_cfg.voxel_length,
                                                            global_config->instance_cfg.sdf_trunc, 
                                                            open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
    
    geometry::Image depth, color;
    int prev_frame_id = -100;
    int prev_save_frame = -100;
    fmfusion::TicTocSequence tic_toc_seq("# Load Integration Export", 3);

    for(int k=0;k<rgbd_table.size();k++){
        RGBDFrameDirs frame_dirs = rgbd_table[k];
        std::string frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
        frame_name = frame_name.substr(0,frame_name.find_last_of("."));
        int frame_id;
        if(global_config->dataset==fmfusion::Config::DATASET_TYPE::REALSENSE || global_config->dataset==fmfusion::Config::DATASET_TYPE::SCANNET)
            frame_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));
        else if(global_config->dataset==fmfusion::Config::DATASET_TYPE::RIO){
            frame_name = frame_name.substr(0,12);
            frame_id = stoi(frame_name.substr(6,12));
        }
        else 
            frame_id = stoi(frame_name);

        if(frame_id>max_frames) break;
        if((frame_id-prev_frame_id)<frame_gap) continue;
        tic_toc_seq.tic();

        utility::LogInfo("Processing frame {:s} ...", frame_name);

        io::ReadImage(frame_dirs.second, depth);
        io::ReadImage(frame_dirs.first, color);

        auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(color, depth, global_config->mapping_cfg.depth_scale, global_config->mapping_cfg.depth_max, false);
        
        std::vector<fmfusion::DetectionPtr> detections;
        bool loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, frame_name, 
                                                        global_config->mapping_cfg, 
                                                        global_config->instance_cfg.intrinsic.width_, 
                                                        global_config->instance_cfg.intrinsic.height_,
                                                        detections);
        tic_toc_seq.toc();
        if(!loaded) continue; 
        
        semantic_mapping.integrate(frame_id,rgbd, pose_table[k], detections);
        if(global_tsdf)
            global_volume.Integrate(*rgbd, global_config->instance_cfg.intrinsic, pose_table[k].inverse().cast<double>());
        tic_toc_seq.toc();

        if(save_instances_gap>0 && frame_id>0 && frame_id%save_instances_gap==0){ 
            // Save instance points
            semantic_mapping.extract_point_cloud();
            semantic_mapping.merge_overlap_instances();
            semantic_mapping.merge_overlap_structural_instances();
            semantic_mapping.extract_bounding_boxes();
            semantic_mapping.Save(output_folder+"/"+sequence_name+std::string("_")+std::to_string(frame_id));
            utility::LogWarning("Save sequence at frame {:d}",frame_id);
        }

        if(save_global_map_gap>0 && frame_id>300 && (frame_id - prev_save_frame)>save_global_map_gap){ 
            // Save global points
            semantic_mapping.extract_point_cloud();
            auto global_instance_pcd = semantic_mapping.export_global_pcd(true,0.02);
            global_instance_pcd->colors_.clear();
            global_instance_pcd->normals_.clear();
            open3d::io::WritePointCloudToPLY(scene_output+"/"+frame_name+".ply",*global_instance_pcd,{});
            utility::LogWarning("Save global instance point cloud at frame {:d}",frame_id);
            std::cout<<scene_output<<"\n";
            prev_save_frame = frame_id;
        }

        if(save_frame_instances>0){
            std::vector<Eigen::Vector3d> instance_centroids = semantic_mapping.export_instance_centroids(0);
            std::vector<std::string> instance_annotations = semantic_mapping.export_instance_annotations(0); 
            assert(instance_centroids.size()==instance_annotations.size());
            fmfusion::IO::save_instance_info(instance_centroids,instance_annotations,
                                            root_dir+"/hydra_lcd/"+frame_name+".txt");

        }

        prev_frame_id = frame_id;
    }
    utility::LogWarning("Finished sequence with {:d} frames",rgbd_table.size());
    if(save_global_map_gap>0) return 0;

    // Post-process
    semantic_mapping.extract_point_cloud();
    semantic_mapping.merge_floor();
    semantic_mapping.merge_overlap_instances();
    // semantic_mapping.merge_overlap_structural_instances(true);
    semantic_mapping.extract_bounding_boxes();

    // Visualize
    if(visualize_flag){
        auto geometries = semantic_mapping.get_geometries(true,true);
        open3d::visualization::DrawGeometries(geometries, sequence_name+output_subseq, 1920, 1080);
    }
    // Save
    semantic_mapping.Save(output_folder+"/"+sequence_name+output_subseq);
    if(global_tsdf){
        auto global_mesh=global_volume.ExtractTriangleMesh();
        io::WriteTriangleMesh(output_folder+"/"+sequence_name+output_subseq+"/mesh_o3d.ply",*global_mesh);
        utility::LogWarning("Save global TSDF volume");
    }
    tic_toc_seq.export_data(output_folder+"/"+sequence_name+output_subseq+"/time_records.txt");
    fmfusion::utility::write_config(output_folder+"/"+sequence_name+output_subseq+"/config.txt",*global_config);

    return 0;
}
