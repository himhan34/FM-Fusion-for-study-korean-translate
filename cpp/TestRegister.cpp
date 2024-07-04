#include <iostream>
#include <memory>
#include <vector>
#include "open3d/Open3D.h"

#include "Common.h"
#include "tools/Tools.h"
#include "tools/IO.h"
#include "tools/Eval.h"
#include "tools/Utility.h"
#include "tools/g3reg_api.h"
#include "mapping/SemanticMapping.h"

#include "sgloop/Graph.h"
#include "sgloop/LoopDetector.h"


bool check_file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char *argv[])
{
    using namespace open3d;

    std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
    std::string src_scene = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string ref_scene = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    std::string frame_name = utility::GetProgramOptionAsString(argc, argv, "--frame_name");
    bool visualization = utility::ProgramOptionExists(argc, argv, "--visualization");

    // Global point cloud for visualization.
    // They are extracted after each sequence is finished.
    auto src_pcd = io::CreatePointCloudFromFile(output_folder+"/"+src_scene+"/instance_map.ply");
    auto ref_pcd = io::CreatePointCloudFromFile(output_folder+"/"+ref_scene+"/instance_map.ply");
    std::cout<<"Load ref pcd size: "<<ref_pcd->points_.size()<<std::endl;
    std::cout<<"Load src pcd size: "<<src_pcd->points_.size()<<std::endl;

    // Load hierarchical correspondences
    Eigen::Matrix4d pred_pose;
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids; // Matched centroids
    fmfusion::O3d_Cloud_Ptr corr_src_pcd, corr_ref_pcd; // Matched point cloud
    std::string corr_folder = output_folder+"/"+src_scene+"/"+ref_scene;

    // Coarse matches
    bool load_results = fmfusion::IO::load_match_results(
                    corr_folder+"/"+frame_name+".txt",
                    pred_pose,src_centroids,ref_centroids);
    if(!load_results){
        utility::LogError("Failed to load match results.");
        return 0;
    }

    // Dense matches
    if(check_file_exists(corr_folder+"/"+frame_name+"_csrc.ply") &&
        check_file_exists(corr_folder+"/"+frame_name+"_cref.ply")){
        std::cout<<"Load dense correspondences."<<std::endl;
        corr_src_pcd = io::CreatePointCloudFromFile(corr_folder+"/"+frame_name+"_csrc.ply");
        corr_ref_pcd = io::CreatePointCloudFromFile(corr_folder+"/"+frame_name+"_cref.ply");

        std::cout<<"Load dense corrs, src: "<<corr_src_pcd->points_.size()
                            <<" ref: "<<corr_ref_pcd->points_.size()<<std::endl;

    }
    else{
        std::cout<<"This is a coarse loop frame. No dense correspondences."<<std::endl;
    }


    // fmfusion::O3d_Cloud_Ptr src_pcd_dense = io::CreatePointCloudFromFile(
    //     corr_folder+"/"+frame_name+"_csrc.ply");

    if(visualization){
        visualization::DrawGeometries({ref_pcd},"TestRegistration",1920,1080);
    }

    return 0;
}

