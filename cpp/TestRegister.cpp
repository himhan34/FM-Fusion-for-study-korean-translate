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


bool check_file_exists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    // The output folder saves online registration results
    std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
    std::string gt_folder = utility::GetProgramOptionAsString(argc, argv, "--gt_folder");
    std::string src_scene = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string ref_scene = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    std::string export_folder = utility::GetProgramOptionAsString(argc, argv, "--export_folder");
    std::string frame_name = utility::GetProgramOptionAsString(argc, argv, "--frame_name");
    bool visualization = utility::ProgramOptionExists(argc, argv, "--visualization");

    // Global point cloud for visualization.
    // They are extracted after each sequence is finished.
    auto recons_src_pcd = io::CreatePointCloudFromFile(output_folder + "/" + src_scene + "/instance_map.ply");
    auto recons_ref_pcd = io::CreatePointCloudFromFile(output_folder + "/" + ref_scene + "/instance_map.ply");
    std::cout << "Load ref pcd size: " << recons_ref_pcd->points_.size() << std::endl;
    std::cout << "Load src pcd size: " << recons_src_pcd->points_.size() << std::endl;

    // Load gt
    Eigen::Matrix4d gt_pose; // T_ref_src
    bool read_gt = fmfusion::IO::read_transformation(gt_folder + "/" + src_scene + "-" + ref_scene + ".txt", gt_pose);
    assert(read_gt);

    // Load hierarchical correspondences
    Eigen::Matrix4d pred_pose;
    std::vector<std::pair<uint32_t, uint32_t>> match_pairs; // Matched instances
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids; // Matched centroids
    fmfusion::O3d_Cloud_Ptr corr_src_pcd(new fmfusion::O3d_Cloud()), corr_ref_pcd(new fmfusion::O3d_Cloud());
    std::string corr_folder = output_folder + "/" + src_scene + "/" + ref_scene;

    // Coarse matches
    bool load_results = fmfusion::IO::load_match_results(
            corr_folder + "/" + frame_name + ".txt",
            pred_pose, match_pairs,
            src_centroids, ref_centroids,
            false);
    int M = match_pairs.size();
    if (!load_results) {
        utility::LogError("Failed to load match results.");
        return 0;
    }

    // Dense matches
    if (check_file_exists(corr_folder + "/" + frame_name + "_csrc.ply") &&
        check_file_exists(corr_folder + "/" + frame_name + "_cref.ply")) {
        std::cout << "Load dense correspondences." << std::endl;
        corr_src_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + frame_name + "_csrc.ply");
        corr_ref_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + frame_name + "_cref.ply");

        std::vector<int> corr_match_indces;
        fmfusion::IO::load_corrs_match_indices(corr_folder + "/" + frame_name + "_cmatches.txt",
                                               corr_match_indces);
        int C = corr_src_pcd->points_.size();
        assert(C == corr_ref_pcd->points_.size());
        assert(C == corr_match_indces.size());

        std::cout << "Load dense corrs, src: " << corr_src_pcd->points_.size()
                  << " ref: " << corr_ref_pcd->points_.size() << std::endl;

        // Demonstrate how to access the node index of each correpondence
        for (int i = 0; i < C; i++) {
            int match_index = corr_match_indces[i];
            assert(match_index < M);
            int src_node_id = match_pairs[match_index].first;
            int ref_node_id = match_pairs[match_index].second;
        }

    } else {
        std::cout << "This is a coarse loop frame. No dense correspondences." << std::endl;
    }

    // Refine the registration
    fmfusion::o3d_utility::Timer timer;
    g3reg::Config config;
//    noise bound的取值
    config.set_noise_bounds({0.2, 0.3});
//    位姿求解优化器的类型
    config.tf_solver = "quatro";
//    基于点到平面的距离做验证
    config.verify_mtd = "plane_based";
    G3RegAPI g3reg(config);
    timer.Start();
    std::vector<float> corr_scores_vec = std::vector<float>(corr_src_pcd->points_.size(), 1.0);
    g3reg.estimate_pose(src_centroids,
                        ref_centroids,
                        corr_src_pcd->points_,
                        corr_ref_pcd->points_,
                        corr_scores_vec,
                        recons_src_pcd,
                        recons_ref_pcd,
                        pred_pose);
    timer.Stop();
    double g3reg_time = timer.GetDurationInMillisecond();
    timer.Start();
    std::cout << "Continue to use ICP to refine the pose\n";
    pred_pose = g3reg.icp_refine(recons_src_pcd, recons_ref_pcd, pred_pose, 0.5, 0.1);
    timer.Stop();
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() + g3reg_time
              << " ms, G3Reg time: " << std::fixed << std::setprecision(3) << g3reg_time << " ms, ICP time: "
              << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() << " ms\n";


    // This is the point at the specific loop frame
    fmfusion::O3d_Cloud_Ptr src_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + frame_name + "_src.ply");

    if (visualization) {
        visualization::DrawGeometries({recons_ref_pcd}, "TestRegistration", 1920, 1080);
    }

    {
        std::cout << "Export results to " << export_folder << std::endl;
        fmfusion::IO::save_pose(export_folder + "/" + frame_name + "_newpose.txt", pred_pose);

        io::WritePointCloudToPLY(export_folder + "/" + frame_name + "_src.ply", *src_pcd, {});
        if (corr_src_pcd && corr_ref_pcd) {
            io::WritePointCloud(export_folder + "/" + frame_name + "_csrc.ply", *corr_src_pcd);
            io::WritePointCloud(export_folder + "/" + frame_name + "_cref.ply", *corr_ref_pcd);
        }
    }

    return 0;
}

