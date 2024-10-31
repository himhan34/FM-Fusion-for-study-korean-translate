// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <vector>
#include <fstream>

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"

#include "tools/Utility.h"

std::vector<std::string> 
  split_str(const std::string s, const std::string delim) 
{
    std::vector<std::string> list;
    auto start = 0U;
    auto end = s.find(delim);
    while (true) {
        list.push_back(s.substr(start, end - start));
        if (end == std::string::npos)
            break;
        start = end + delim.length();
        end = s.find(delim, start);
    }
    return list;
}

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
            auto parts = split_str(line,"=");
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

bool read_rio_intrinsic(const std::string intrinsic_dir, 
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
            // open3d::utility::LogInfo("{:s}",line);
            auto parts = split_str(line," ");
            if(parts[0].find("colorWidth")!=string::npos)
                width = stoi(parts[2]);
            else if (parts[0].find("colorHeight")!=string::npos)
                height = stoi(parts[2]);
            else if (parts[0].find("ColorIntrinsic")!=string::npos){
                fx = stod(parts[2]);
                fy = stod(parts[7]);
                cx = stod(parts[4]);
                cy = stod(parts[8]);
            }
                // utility::LogInfo("Intrinsic: {:s}",parts[1].substr(1));
        }
        file.close();
        // std::cout<<width<<","<<height<<"\n";
        // std::cout<<fx<<","<<fy<<","<<cx<<","<<cy<<","<<"\n";
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
        auto row0 = split_str(line0," ");
        auto row1 = split_str(line1," ");
        fx = stod(row0[0]); cx = stod(row0[2]);
        fy = stod(row1[1]); cy = stod(row1[2]);

        getline(f_shape,sline0);
        getline(f_shape,sline1);
        getline(f_shape,sline2);
        getline(f_shape,sline3);

        width = stoi(split_str(sline2,":")[1]);
        height = stoi(split_str(sline3,":")[1]);

        f_mat.close();
        f_shape.close();

        // std::cout<<width<<","<<height<<"\n";
        // std::cout<<fx<<","<<fy<<","<<cx<<","<<cy<<","<<"\n";
        intrinsic_.SetIntrinsics(width,height,fx,fy,cx,cy);
        return true;
    }
    else return false;

}

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > IntegrateRGBD [options]");
    utility::LogInfo("      Integrate RGBD stream and extract geometry.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --root, -h                : Set dataset root.");
    utility::LogInfo("    --depth_scale             : Default:1000.0, TUM:5000.0");
    utility::LogInfo("    --max_frames              : Default:1000. Only the first MAX_FRAMES are integrated.");
//     utility::LogInfo("    --match file              : The match file of an RGBD stream. Must have.");
//     utility::LogInfo("    --log file                : The log trajectory file. Must have.");
    utility::LogInfo("    --save_pointcloud         : Save a point cloud created by marching cubes.");
    utility::LogInfo("    --save_mesh               : Save a mesh created by marching cubes.");
    utility::LogInfo("    --save_voxel              : Save a point cloud of the TSDF voxel.");
    utility::LogInfo("    --every_k_frames k        : Save/reset every k frames. Default: 0 (none).");
    utility::LogInfo("    --length l                : Length of the volume, in meters. Default: 4.0.");
    utility::LogInfo("    --resolution r            : Resolution of the voxel grid. Default: 512.");
    utility::LogInfo("    --sdf_trunc_percentage t  : TSDF truncation percentage, of the volume length. Default: 0.01.");
    utility::LogInfo("    --visualization           : Visualize the result.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc <= 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string root_dir =
            utility::GetProgramOptionAsString(argc, argv, "--root");
    std::string association_name = 
        utility::GetProgramOptionAsString(argc, argv, "--association","/data_association.txt");
    std::string trajectory_name = 
        utility::GetProgramOptionAsString(argc, argv, "--trajectory","/trajectory.log");
    std::string intrinsic_filename =
            utility::GetProgramOptionAsString(argc, argv, "--intrinsic");
    std::string output_dir =
            utility::GetProgramOptionAsString(argc, argv, "--output","");
    int max_frames =
            utility::GetProgramOptionAsInt(argc, argv, "--max_frames", 1000);
    bool save_pointcloud =
            utility::ProgramOptionExists(argc, argv, "--save_pointcloud");
    bool save_mesh = utility::ProgramOptionExists(argc, argv, "--save_mesh");
    bool save_voxel = utility::ProgramOptionExists(argc, argv, "--save_voxel");
    double depth_scale = utility::GetProgramOptionAsDouble(argc,argv,"--depth_scale", 1000.0);
    int every_k_frames =
            utility::GetProgramOptionAsInt(argc, argv, "--every_k_frames", 0);
    double length =
            utility::GetProgramOptionAsDouble(argc, argv, "--length", 4.0);
    double max_depth = utility::GetProgramOptionAsDouble(argc, argv, "--max_depth", 4.0);
    int resolution =
            utility::GetProgramOptionAsInt(argc, argv, "--resolution", 512);
    double sdf_trunc_percentage = utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc_percentage", 0.01);
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    bool visualize_flag = utility::ProgramOptionExists(argc, argv, "--visualization");

    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    utility::LogInfo("Read RGBD sequence from {:s}", root_dir);

    // Read the match file and the trajectory file.
    std::string match_filename = root_dir+association_name; //"/data_association.txt";
    std::string trajectory_dir = root_dir+trajectory_name; //"/trajectory.log";
    auto camera_trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_dir);
    std::string dir_name =
            utility::filesystem::GetFileParentDirectory(match_filename).c_str();
    FILE *file = utility::filesystem::FOpen(match_filename, "r");
    if (file == NULL) {
        utility::LogWarning("Unable to open file {}", match_filename);
        fclose(file);
        return 0;
    }
    char buffer[DEFAULT_IO_BUFFER_SIZE];
    int index = 0;
    int save_index = 0;
    pipelines::integration::ScalableTSDFVolume volume(
            length / (double)resolution, length * sdf_trunc_percentage,
            pipelines::integration::TSDFVolumeColorType::RGB8);

    utility::FPSTimer timer("Process RGBD stream",
                            (int)camera_trajectory->parameters_.size());
    geometry::Image depth, color;
    camera::PinholeCameraIntrinsic intrinsic_;

    // Read the intrinsic file
    if(intrinsic_filename!=""){
        using namespace std;
        read_scannet_intrinsic(root_dir+"/"+ intrinsic_filename,intrinsic_);
    }
    else
        read_rs_intrinsic(root_dir+"/intrinsic.txt",intrinsic_);

    visualization::Visualizer vis;
    if (visualize_flag){
        vis.CreateVisualizerWindow("Open3D", 1600, 900);
    }

    while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file) && index<max_frames) {
        std::vector<std::string> st = utility::SplitString(buffer, "\t\r\n ");

        if (st.size() >= 2) {
            // if(index % 10==0) 
            std::string frame_name = st[0].substr(st[0].find_last_of("/")+1,st[0].size()); // eg. frame-000000.png
            utility::LogInfo("Processing frame {:s},{:s} ...", st[0], st[1]);

            io::ReadImage(dir_name + st[0], depth);
            io::ReadImage(dir_name + st[1], color);
            // utility::LogInfo("Read depth {:d}x{:d}, color {:d}x{:d}", depth.width_, depth.height_, color.width_, color.height_);

            auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
                    color, depth, depth_scale, max_depth, false);
            if (index == 0 ||
                (every_k_frames > 0 && index % every_k_frames == 0)) {
                volume.Reset();
            }

            volume.Integrate(*rgbd,
                            intrinsic_,
                             camera_trajectory->parameters_[index].extrinsic_);

            if (visualize_flag && index%10==0){
                cv::Mat depth_vis, color_vis;

                depth_vis = cv::imread(dir_name + st[0],cv::IMREAD_ANYDEPTH);
                color_vis = cv::imread(dir_name + st[1],cv::IMREAD_COLOR);
                auto pcd = volume.ExtractPointCloud();
                vis.AddGeometry(pcd);
                vis.UpdateGeometry();
                vis.PollEvents();
                vis.UpdateRender();
                
                cv::Mat depth_vis_color;
                depth_vis.convertTo(depth_vis_color,CV_8UC1,255.0/1000.0);
                cv::applyColorMap(depth_vis_color,depth_vis_color,cv::COLORMAP_JET);

                cv::imshow("depth",depth_vis_color);
                cv::waitKey(10);
            }
        
            if (index%10==0){ // query observed points
                auto observed_cloud = std::make_shared<geometry::PointCloud>();
                // if (volume.query_observed_points(depth_cloud, observed_cloud)){
                //     utility::Timer timer;
                //     timer.Start();
                //     auto instance_uv = fmfusion::utility::PrjectionCloudToDepth(*observed_cloud,
                //         camera_trajectory->parameters_[index].extrinsic_,intrinsic_, 3);
                //     timer.Stop();
                //     utility::LogInfo("Frame {:d}, points:{:d}, Projection time: {:f} ms", 
                //         index, observed_cloud->points_.size(), timer.GetDurationInMillisecond());
                //     cv::applyColorMap(*instance_uv,*instance_uv,cv::COLORMAP_JET);
                //     cv::imwrite(root_dir+"/tmp/"+frame_name,*instance_uv);
                // }
                // else
                //     utility::LogInfo("Frame {:d} No observed points", index);
            }

            index++;
            timer.Signal();

        }
    }

    
    {
        if(output_dir.size()<1) output_dir = root_dir;
        struct stat buffer;
        utility::LogInfo("Saving fragment {:s} ...", output_dir);

        if (save_pointcloud) {
                utility::LogInfo("Saving pointcloud {:d} ...", save_index);
                auto pcd = volume.ExtractPointCloud();
                io::WritePointCloud(output_dir+"/pointcloud_o3d.ply",
                                *pcd);
        }
        if (save_mesh) {
                auto mesh = volume.ExtractTriangleMesh();
                utility::LogInfo("Saving mesh with {:d} points", mesh->vertices_.size());
                io::WriteTriangleMesh(output_dir+"/mesh_o3d.ply",
                                        *mesh);                
        }
        save_index++;
    }

    fclose(file);
    return 0;
}
