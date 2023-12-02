#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"

#include "Utility.h"
#include "SceneGraph.h"


typedef std::pair<std::string, std::string> RGBDFrameDirs;

bool read_rs_intrinsic(const std::string intrinsic_dir, 
    open3d::camera::PinholeCameraIntrinsic &intrinsic_)
{
    using namespace std;
    open3d::utility::LogInfo("Read intrinsic {:s}",intrinsic_dir);
    fstream file(intrinsic_dir,fstream::in);
    double fx,fy,cx,cy;
    int width, height;
    if(file.is_open()){
        string line{""};
        while(getline(file,line)){
            open3d::utility::LogInfo("{:s}",line);
            auto parts = fmfusion::utility::split_str(line,"=");
            if(parts[0].find("color_width")!=string::npos)
                width = stoi(parts[1].substr(1));
            else if (parts[0].find("color_height")!=string::npos)
                height = stoi(parts[1].substr(1));
            else if (parts[0].find("color_fx")!=string::npos)
                fx = stod(parts[1].substr(1));
            else if ((parts[0].find("color_fy")!=string::npos))
                fy = stod(parts[1].substr(1));
            else if ((parts[0].find("color_cx")!=string::npos))
                cx = stod(parts[1].substr(1));
            else if ((parts[0].find("color_cy")!=string::npos))
                cy = stod(parts[1].substr(1));
                // utility::LogInfo("Intrinsic: {:s}",parts[1].substr(1));
        }
        file.close();
        std::cout<<fx<<","<<fy<<"\n";
        // open3d::utility::LogInfo("{:d},{:d},{:d},{:d}",fx,fy,cx,cy);
        intrinsic_.SetIntrinsics(width,height,fx,fy,cx,cy);
        return true;
    }
    return false;
}

bool read_scannet_intrinsic(const std::string intrinsic_folder,
    open3d::camera::PinholeCameraIntrinsic &intrinsic_)
{
    using namespace std;
    open3d::utility::LogInfo("Read intrinsic {:s}",intrinsic_folder);
    fstream f_mat(intrinsic_folder+"/intrinsic_depth.txt",fstream::in);
    fstream f_shape(intrinsic_folder+"/sensor_shapes.txt",fstream::in);
    double fx,fy,cx,cy;
    int width, height;
    if(f_mat.is_open() && f_shape.is_open()){

        string line{""};
        string line0, line1; 
        string sline0,sline1,sline2, sline3;
        getline(f_mat,line0);
        getline(f_mat,line1);
        auto row0 = fmfusion::utility::split_str(line0," ");
        auto row1 = fmfusion::utility::split_str(line1," ");
        fx = stod(row0[0]); cx = stod(row0[2]);
        fy = stod(row1[1]); cy = stod(row1[2]);

        getline(f_shape,sline0);
        getline(f_shape,sline1);
        getline(f_shape,sline2);
        getline(f_shape,sline3);

        width = stoi(fmfusion::utility::split_str(sline2,":")[1]);
        height = stoi(fmfusion::utility::split_str(sline3,":")[1]);

        f_mat.close();
        f_shape.close();

        // std::cout<<width<<","<<height<<"\n";
        // std::cout<<fx<<","<<fy<<","<<cx<<","<<cy<<","<<"\n";
        intrinsic_.SetIntrinsics(width,height,fx,fy,cx,cy);
        return true;
    }
    else return false;

}

// sort rgb frames    
bool frames_srt_func(const std::string &a, const std::string &b){
    std::string name_a = a.substr(a.find_last_of("/")+1); 
    std::string name_b = b.substr(b.find_last_of("/")+1); // frame-000000.png
    if(name_a.find("-")!=std::string::npos) 
        return stoi(name_a.substr(name_a.find_last_of("-")+1,name_a.find_last_of("."))) < stoi(name_b.substr(name_b.find_last_of("-")+1,name_b.find_last_of(".")));
    else
        return stoi(name_a.substr(0,name_a.find_last_of("."))) < stoi(name_b.substr(0,name_b.find_last_of(".")));
};

void construct_sorted_frame_table(const std::string &scene_dir,
    std::vector<std::string> &rgb_frames, std::vector<std::string> &depth_frames,
    std::vector<RGBDFrameDirs> &frame_table, std::vector<Eigen::Matrix4d> &pose_table)
{
    using namespace std;
    using namespace open3d::utility::filesystem;

    assert(rgb_frames.size()==depth_frames.size()),"rgb and depth frames should have the same size.";
    int K = rgb_frames.size();

    std::sort(rgb_frames.begin(),rgb_frames.end(),frames_srt_func);
    std::sort(depth_frames.begin(),depth_frames.end(),frames_srt_func);

    for(int k=0;k<K;k++){
        auto frame_name = rgb_frames[k].substr(rgb_frames[k].find_last_of("/")+1); // eg. frame-000000.png
        string pose_dir = JoinPath({scene_dir,"pose",frame_name.substr(0,frame_name.find_last_of("."))+".txt"});
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        fstream file(pose_dir,fstream::in);
        if(file.is_open()){
            string line{""};
            int i=0;
            while(getline(file,line)){
                // std::cout<<line<<"\n";
                std::stringstream ss(line);
                int j=0;
                while(ss.good() && j<4){
                    std::string substr;
                    getline(ss, substr, ' ');
                    if(substr.empty()) continue;
                    // cout<<substr<<",";
                    pose(i,j) = stod(substr);
                    j++;
                }
                // std::cout<<"\n";
                i++;
            }
            // std::cout<<pose<<"\n";
            file.close();
        }
        else{
            open3d::utility::LogWarning("Failed to read pose file {:s}",pose_dir);
            continue;
        }

        // cout<<rgb_frames[k]<<endl;
        // cout<<depth_frames[k]<<endl;
        // cout<<pose<<endl;

        frame_table.emplace_back(rgb_frames[k],depth_frames[k]);
        pose_table.emplace_back(pose);
    }

    open3d::utility::LogInfo("Read {:d} frames",frame_table.size());

}

int main(int argc, char *argv[]) 
{
    using namespace open3d;

    std::string config_file = 
            utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string root_dir =
            utility::GetProgramOptionAsString(argc, argv, "--root");
    std::string association_name = 
            utility::GetProgramOptionAsString(argc, argv, "--association");
    std::string prediction_folder = 
            utility::GetProgramOptionAsString(argc, argv, "--prediction");
    std::string output_folder = 
            utility::GetProgramOptionAsString(argc, argv, "--output","./");
    int begin_frames = 
            utility::GetProgramOptionAsInt(argc, argv, "--begin_frames", 0);
    int max_frames =
            utility::GetProgramOptionAsInt(argc, argv, "--max_frames", 5000);
    int verbose = 
            utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    bool visualize_flag = 
            utility::ProgramOptionExists(argc, argv, "--visualization");
    int frame_gap = 
            utility::GetProgramOptionAsInt(argc, argv, "--frame_gap", 1);

    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    utility::LogInfo("Read configuration from {:s}",config_file);
    utility::LogInfo("Read RGBD sequence from {:s}", root_dir);
    std::string sequence_name = *utility::filesystem::GetPathComponents(root_dir).rbegin();

    // Read the match file and the trajectory file.

    // Load frames information
    std::vector<RGBDFrameDirs> rgb_table;
    std::vector<Eigen::Matrix4d> pose_table;
    if(association_name.empty()) {
        utility::LogInfo("--- Read all RGB-D frames in {:s} ---",root_dir);
        std::vector<std::string> depth_files, rgb_files;
        utility::filesystem::ListFilesInDirectory(root_dir+"/color", rgb_files);
        utility::filesystem::ListFilesInDirectory(root_dir+"/depth", depth_files);
        construct_sorted_frame_table(root_dir,rgb_files,depth_files,rgb_table,pose_table);
        utility::LogInfo("find {:d} rgb-d frame with pose",rgb_table.size());
        if(rgb_table.size()>4000) return 0;
    }
    else{//todo
        // auto camera_trajectory =
        //         io::CreatePinholeCameraTrajectoryFromFile(trajectory_dir);
    }


    utility::FPSTimer timer("Process RGBD stream", (int)rgb_table.size());
    geometry::Image depth, color;

    // Visualization (optional)
    visualization::Visualizer vis;
    if (visualize_flag) vis.CreateVisualizerWindow("Open3D", 1600, 900);
    
    //
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
    if(sg_config->tmp_dir.empty())
        sg_config->tmp_dir = output_folder;
    if(sg_config==nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }

    //
    std::string rgb_folder;
    switch (sg_config->dataset)
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
    }

    //
    std::array<std::string,2> sub_sequences = {"a","b"};
    int sub_sequence_length = rgb_table.size()/2;
    int subseq_begin = 0;
    int subseq_end = sub_sequence_length;

    for(std::string subseq:sub_sequences){
        // Start integration
        fmfusion::SceneGraph scene_graph(*sg_config);
        int prev_frame_id = -100;
        for(int k=subseq_begin;k<subseq_end;k++){
            RGBDFrameDirs frame_dirs = rgb_table[k];
            std::string frame_name = frame_dirs.first.substr(frame_dirs.first.find_last_of("/")+1); // eg. frame-000000.png
            frame_name = frame_name.substr(0,frame_name.find_last_of("."));
            int frame_id;
            if(sg_config->dataset==fmfusion::Config::DATASET_TYPE::REALSENSE || sg_config->dataset==fmfusion::Config::DATASET_TYPE::SCANNET)
                frame_id = stoi(frame_name.substr(frame_name.find_last_of("-")+1));
            else 
                frame_id = stoi(frame_name);
            
            if(frame_id>max_frames) break;
            if((frame_id-prev_frame_id)<frame_gap) continue;
            
            utility::LogInfo("Processing frame {:s} ...", frame_name);

            io::ReadImage(frame_dirs.second, depth);
            io::ReadImage(frame_dirs.first, color);

            auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(color, depth, sg_config->depth_scale, sg_config->depth_max, false);
            
            std::vector<fmfusion::DetectionPtr> detections;
            bool loaded = fmfusion::utility::LoadPredictions(root_dir+'/'+prediction_folder, frame_name, *sg_config, detections);
            if(!loaded) continue; 
            
            scene_graph.integrate(frame_id,rgbd, pose_table[k], detections);
            prev_frame_id = frame_id;
        
        }
        utility::LogWarning("Finished sequence");

        // Post-process
        scene_graph.extract_point_cloud();
        scene_graph.merge_overlap_instances();
        scene_graph.extract_bounding_boxes();
        scene_graph.merge_overlap_structural_instances();
        
        // Visualize
        // auto geometries = scene_graph.get_geometries(true,true);
        // open3d::visualization::DrawGeometries(geometries, sequence_name+subseq, 1920, 1080);

        // Save
        scene_graph.Save(output_folder+"/"+sequence_name+subseq);
        subseq_begin += sub_sequence_length;
        subseq_end   += sub_sequence_length;

    }
    return 0;
}