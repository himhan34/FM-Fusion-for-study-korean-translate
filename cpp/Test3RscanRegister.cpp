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

double eval_registration_error(const fmfusion::O3d_Cloud_Ptr &src_cloud, const Eigen::Matrix4d &pred_tf,
                               const Eigen::Matrix4d &gt_tf) {
    Eigen::Matrix4d realignment_transform = gt_tf.inverse() * pred_tf;
    double rmse = 0.0;
    for (int i = 0; i < src_cloud->points_.size(); i++) {
        Eigen::Vector3d src_point = src_cloud->points_[i];
        Eigen::Vector3d realigned_src_point =
                realignment_transform.block<3, 3>(0, 0) * src_point + realignment_transform.block<3, 1>(0, 3);
        rmse += (realigned_src_point - src_cloud->points_[i]).norm();
    }
    return rmse / src_cloud->points_.size();
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    // The output folder saves online registration results
    std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string corr_folder = utility::GetProgramOptionAsString(argc, argv, "--corr_folder");
    std::string gt_folder = utility::GetProgramOptionAsString(argc, argv, "--gt_folder");
    std::string src_scene = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string ref_scene = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    bool enable_icp = utility::ProgramOptionExists(argc, argv, "--enable_icp");

    // registration parameters
    double verify_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--verify_voxel", 0.5);
    double search_radius = utility::GetProgramOptionAsDouble(argc, argv, "--search_radius", 0.5);
    double icp_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--icp_voxel", 0.2);
    double ds_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--ds_voxel", 0.5);
    int ds_num = utility::GetProgramOptionAsInt(argc, argv, "--ds_num", 9);
    double nms_thd = utility::GetProgramOptionAsDouble(argc, argv, "--nms_thd", 0.05);
    double inlier_threshold = utility::GetProgramOptionAsDouble(argc, argv, "--inlier_threshold", 0.3);
    int max_corr_number = utility::GetProgramOptionAsInt(argc, argv, "--max_corr_number", 1000);
    // std::string tf_solver_type = utility::GetProgramOptionAsString(argc, argv, "--tf_solver", "quatro");

    bool visualization = utility::ProgramOptionExists(argc, argv, "--visualization");
//    std::cout << "Src scene: " << src_scene << ", Ref scene: " << ref_scene << std::endl;

    // Global point cloud for visualization.
    auto src_pcd = io::CreatePointCloudFromFile(corr_folder + "/src_instances.ply");
    auto ref_pcd = io::CreatePointCloudFromFile(corr_folder + "/ref_instances.ply");
//    std::cout << "Load ref pcd size: " << ref_pcd->points_.size() << std::endl;
//    std::cout << "Load src pcd size: " << src_pcd->points_.size() << std::endl;

    // Load reconstructed scene graph
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, false);
    if (sg_config == nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }

    // Load gt
    Eigen::Matrix4d gt_pose; // T_ref_src
    bool read_gt = fmfusion::IO::read_transformation(gt_folder + "/" + src_scene + "-" + ref_scene + ".txt", gt_pose);
    assert(read_gt);

    // Load hierarchical correspondences
    Eigen::Matrix4d pred_pose;
    std::vector<std::pair<uint32_t, uint32_t>> match_pairs; // Matched instances id
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids; // Matched centroids
    fmfusion::O3d_Cloud_Ptr corr_src_pcd(new fmfusion::O3d_Cloud()), corr_ref_pcd(new fmfusion::O3d_Cloud());

    // Load coarse matches
    float ref_frame_timestamp;
    bool load_results;
    std::vector<bool> match_tp_masks;
    load_results = fmfusion::IO::load_node_matches(corr_folder + "/node_matches.txt",
                                                   match_pairs,
                                                   match_tp_masks,
                                                   src_centroids, ref_centroids,
                                                   false);
    if (!load_results) {
        utility::LogError("Failed to load match results.");
        return 0;
    }
    int M = match_pairs.size();
//    std::cout << "Load match pairs: " << match_pairs.size() << std::endl;

    // Load Dense matches
    std::vector<float> corr_scores_vec;
    corr_src_pcd = io::CreatePointCloudFromFile(corr_folder + "/corr_src.ply");
    corr_ref_pcd = io::CreatePointCloudFromFile(corr_folder + "/corr_ref.ply");

    int C = corr_src_pcd->points_.size();
    assert(C == corr_ref_pcd->points_.size());
//    std::cout << "Load " << corr_src_pcd->points_.size() << " dense corrs" << std::endl;
    corr_scores_vec = std::vector<float>(C, 0.5);
    downsample_corr_nms(corr_src_pcd->points_, corr_ref_pcd->points_, corr_scores_vec, nms_thd);
    downsample_corr_topk(corr_src_pcd->points_, corr_ref_pcd->points_, corr_scores_vec, ds_voxel, max_corr_number);

    // Refine the registration
    fmfusion::o3d_utility::Timer timer;
    G3RegAPI::Config config;
//    noise bound的取值
    config.set_noise_bounds({0.2, 0.3});
//    位姿求解优化器的类型
    config.nms_thd = nms_thd;
    config.ds_num = ds_num;
    config.plane_resolution = verify_voxel;
    config.max_corr_num = max_corr_number;
//    基于点到平面的距离做验证
    config.verify_mtd = "plane_based";
    config.search_radius = search_radius;
//    ICP
    config.icp_voxel = icp_voxel;
    config.ds_voxel = ds_voxel;

    G3RegAPI g3reg(config);
    timer.Start();
    // g3reg.analyze_correspondences(src_centroids,
    //                                             ref_centroids,
    //                                             corr_src_pcd->points_,
    //                                             corr_ref_pcd->points_,
    //                                             ins_corr_map, gt_pose);

    if (corr_src_pcd->points_.size() > 0) {
        // dense registration
        double inlier_ratio = g3reg.estimate_pose_gnc(src_centroids, ref_centroids, corr_src_pcd->points_,
                                                      corr_ref_pcd->points_, corr_scores_vec);
        std::cout << "Inlier ratio: " << inlier_ratio << std::endl;
        double true_inlier_ratio = g3reg.compute_true_inlier_ratio(src_centroids, ref_centroids, corr_src_pcd->points_,
                                                                   corr_ref_pcd->points_, gt_pose);
//        inlier_ratio = 0.0;
        if (inlier_ratio < inlier_threshold) {
            g3reg.estimate_pose(src_centroids, ref_centroids, corr_src_pcd->points_, corr_ref_pcd->points_,
                                corr_scores_vec, src_pcd, ref_pcd);
        }
    } else { // not expected
        assert(false);
    }

    std::vector<Eigen::Matrix4d> candidates = g3reg.reg_result.candidates;
    pred_pose = g3reg.reg_result.tf;
    timer.Stop();
    double g3reg_time = timer.GetDurationInMillisecond();
    timer.Start();
    if (enable_icp) {
        std::cout << "Continue to use ICP to refine the pose\n";
        pred_pose = g3reg.icp_refine(src_pcd, ref_pcd, pred_pose);
    }

    timer.Stop();
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() + g3reg_time
              << " ms, G3Reg time: " << std::fixed << std::setprecision(3) << g3reg_time << " ms, ICP time: "
              << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() << " ms\n";
    std::cout << "RMSE: " << eval_registration_error(src_pcd, pred_pose, gt_pose) << std::endl;

    if (visualization) {
        src_pcd->Transform(pred_pose);
        src_pcd->PaintUniformColor(Eigen::Vector3d(0.0, 0.651, 0.929));
        ref_pcd->PaintUniformColor(Eigen::Vector3d(1.0, 0.706, 0));
        visualization::DrawGeometries({src_pcd, ref_pcd}, "TestRegistration", 1920, 1080);
    }

    {
        std::cout << "Export results to " << corr_folder << std::endl;
        fmfusion::IO::save_pose(corr_folder + "/pred_newpose.txt", pred_pose);
    }

    return 0;
}

