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
    std::string src_frame_name = utility::GetProgramOptionAsString(argc, argv, "--frame_name");
    std::string ref_frame_map_dir = utility::GetProgramOptionAsString(argc, argv, "--ref_frame_map_dir");
    std::string gt_folder = utility::GetProgramOptionAsString(argc, argv, "--gt_folder");
    std::string src_scene = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string ref_scene = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    std::string export_folder = utility::GetProgramOptionAsString(argc, argv, "--export_folder");

    // registration parameters
    double verify_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--verify_voxel", 0.5);
    double search_radius = utility::GetProgramOptionAsDouble(argc, argv, "--search_radius", 0.5);
    double icp_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--icp_voxel", 0.2);
    double ds_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--ds_voxel", 0.5);
    int ds_num = utility::GetProgramOptionAsInt(argc, argv, "--ds_num", 9);

    bool visualization = utility::ProgramOptionExists(argc, argv, "--visualization");

    std::string corr_folder = output_folder + "/" + src_scene + "/" + ref_scene;

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

    // Coarse matches
    bool load_results = fmfusion::IO::load_match_results(
            corr_folder + "/" + src_frame_name + ".txt",
            pred_pose, match_pairs,
            src_centroids, ref_centroids,
            false);
    std::cout << "Load match pairs: " << match_pairs.size() << std::endl;
    std::cout << "Load centroids: " << src_centroids.size() << std::endl;
    int M = match_pairs.size();
    if (!load_results) {
        utility::LogError("Failed to load match results.");
        return 0;
    }

    // Current dense map
    // The reference point cloud is only available in dense mode
    fmfusion::O3d_Cloud_Ptr src_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + src_frame_name + "_src.ply");
    fmfusion::O3d_Cloud_Ptr ref_pcd;

    // Dense matches
    std::vector<float> corr_scores_vec;
    std::map<int, std::vector<int>> ins_corr_map;
    bool dense_mode = false;
    if (check_file_exists(corr_folder + "/" + src_frame_name + "_csrc.ply") &&
        check_file_exists(corr_folder + "/" + src_frame_name + "_cref.ply")) {
        std::cout << "Load dense correspondences." << std::endl;
        dense_mode = true;
        ref_pcd = io::CreatePointCloudFromFile(ref_frame_map_dir);
        std::cout << "load ref_pcd size: " << ref_pcd->points_.size() << std::endl;

        corr_src_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + src_frame_name + "_csrc.ply");
        corr_ref_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + src_frame_name + "_cref.ply");

        std::vector<int> corr_match_indces;
        fmfusion::IO::load_corrs_match_indices(corr_folder + "/" + src_frame_name + "_cmatches.txt",
                                               corr_match_indces,
                                               corr_scores_vec);
        int C = corr_src_pcd->points_.size();
        assert(C == corr_ref_pcd->points_.size());
        assert(C == corr_match_indces.size());

        std::cout << "Load dense corrs, src: " << corr_src_pcd->points_.size()
                  << " ref: " << corr_ref_pcd->points_.size() << std::endl;

        // Demonstrate how to access the node index of each correpondence
        for (int i = 0; i < C; i++) {
            int match_index = corr_match_indces[i];
            assert(match_index < M);
            if (ins_corr_map.find(match_index) == ins_corr_map.end()) {
                ins_corr_map[match_index] = std::vector<int>();
            }
            ins_corr_map[match_index].push_back(i);
        }
    } else {
        std::cout << "This is a coarse loop frame. No dense correspondences." << std::endl;
        ref_pcd = std::make_shared<fmfusion::O3d_Cloud>();
        std::cout << "load ref_pcd size: " << ref_pcd->points_.size() << std::endl;
    }

    // Refine the registration
    fmfusion::o3d_utility::Timer timer;
    G3RegAPI::Config config;
//    noise bound的取值
    config.set_noise_bounds({0.2, 0.3});
//    位姿求解优化器的类型
    config.tf_solver = "quatro"; // gnc quatro
    config.ds_num = ds_num;
    config.plane_resolution = verify_voxel;
//    基于点到平面的距离做验证
    config.verify_mtd = "plane_based";
    config.search_radius = search_radius;
//    ICP
    config.icp_voxel = icp_voxel;
    config.ds_voxel = ds_voxel;

    G3RegAPI g3reg(config);
    timer.Start();
    g3reg.analyze_correspondences(src_centroids,
                                  ref_centroids,
                                  corr_src_pcd->points_,
                                  corr_ref_pcd->points_,
                                  ins_corr_map, gt_pose);

    if (corr_src_pcd->points_.size() > 0) {
        // dense registration
        double inlier_ratio = 0.0;
        g3reg.estimate_pose(src_centroids, ref_centroids, corr_src_pcd->points_, corr_ref_pcd->points_, inlier_ratio);
        if (inlier_ratio < 0.4) {
            g3reg.estimate_pose(src_centroids, ref_centroids, corr_src_pcd->points_, corr_ref_pcd->points_,
                                corr_scores_vec, src_pcd, ref_pcd);
        }
    } else {
        g3reg.estimate_pose(src_centroids, ref_centroids, corr_src_pcd->points_, corr_ref_pcd->points_,
                            corr_scores_vec, src_pcd, ref_pcd);
    }

    std::vector<Eigen::Matrix4d> candidates = g3reg.reg_result.candidates;
    pred_pose = g3reg.reg_result.tf;
    timer.Stop();
    double g3reg_time = timer.GetDurationInMillisecond();
    timer.Start();
    if (dense_mode) {
        std::cout << "Continue to use ICP to refine the pose\n";
        pred_pose = g3reg.icp_refine(src_pcd, ref_pcd, pred_pose);
    }
    timer.Stop();
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() + g3reg_time
              << " ms, G3Reg time: " << std::fixed << std::setprecision(3) << g3reg_time << " ms, ICP time: "
              << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() << " ms\n";


    if (visualization) {
        src_pcd->Transform(pred_pose);
        src_pcd->PaintUniformColor(Eigen::Vector3d(0.0, 0.651, 0.929));
        ref_pcd->PaintUniformColor(Eigen::Vector3d(1.0, 0.706, 0));
        visualization::DrawGeometries({src_pcd, ref_pcd}, "TestRegistration", 1920, 1080);
    }

    {
        std::cout << "Export results to " << export_folder << std::endl;
//        std::cout << "GT euler angle: " << std::endl;
//        std::cout << gt_pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2) * 180 / M_PI << std::endl;
        fmfusion::IO::save_pose(export_folder + "/" + src_frame_name + "_newpose.txt", pred_pose);
//        std::cout << "Diff between pred and GT:" << std::endl;
//        std::cout << (pred_pose * gt_pose.inverse()) << std::endl;
//        for (int i = 0; i < candidates.size(); i++) {
//            std::cout << "Diff between candidate " << i << " and GT:" << std::endl;
//            std::cout << (candidates[i] * gt_pose.inverse()) << std::endl;
//        }
        io::WritePointCloudToPLY(export_folder + "/" + src_frame_name + "_src.ply", *src_pcd, {});
        if (corr_src_pcd->points_.size() > 0 && corr_ref_pcd->points_.size() > 0) {
            io::WritePointCloud(export_folder + "/" + src_frame_name + "_csrc.ply", *corr_src_pcd);
            io::WritePointCloud(export_folder + "/" + src_frame_name + "_cref.ply", *corr_ref_pcd);
        }
    }

    return 0;
}

