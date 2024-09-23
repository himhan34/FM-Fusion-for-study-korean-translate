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

//def eval_registration_error(src_cloud, pred_tf, gt_tf):
//    src_points = np.asarray(src_cloud.points)
//    src_points = torch.from_numpy(src_points).float()
//    pred_tf = torch.from_numpy(pred_tf).float()
//    gt_tf = torch.from_numpy(gt_tf).float()
//
//    realignment_transform = torch.matmul(torch.inverse(gt_tf), pred_tf)
//    realigned_src_points_f = apply_transform(src_points, realignment_transform)
//    rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
//    return rmse

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
    std::string result_folder = utility::GetProgramOptionAsString(argc, argv, "--result_folder");
    std::string src_frame_name = utility::GetProgramOptionAsString(argc, argv, "--frame_name");
    std::string ref_frame_map_dir = utility::GetProgramOptionAsString(argc, argv, "--ref_frame_map_dir");
    std::string gt_folder = utility::GetProgramOptionAsString(argc, argv, "--gt_folder");
    std::string src_scene = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string ref_scene = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    std::string new_result_folder = utility::GetProgramOptionAsString(argc, argv, "--new_result_folder");
    bool enable_icp = utility::ProgramOptionExists(argc, argv, "--enable_icp");

    // registration parameters
    bool downsample_corr = utility::ProgramOptionExists(argc, argv, "--downsample_corr");
    double verify_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--verify_voxel", 0.5);
    double search_radius = utility::GetProgramOptionAsDouble(argc, argv, "--search_radius", 0.5);
    double icp_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--icp_voxel", 0.2);
    double ds_voxel = utility::GetProgramOptionAsDouble(argc, argv, "--ds_voxel", 0.5);
    int ds_num = utility::GetProgramOptionAsInt(argc, argv, "--ds_num", 9);
    double nms_thd = utility::GetProgramOptionAsDouble(argc, argv, "--nms_thd", 0.05);
    double inlier_threshold = utility::GetProgramOptionAsDouble(argc, argv, "--inlier_threshold", 0.3);
    int max_corr_number = utility::GetProgramOptionAsInt(argc, argv, "--max_corr_number", 1000);
    bool visualization = utility::ProgramOptionExists(argc, argv, "--visualization");
    bool register_sg = utility::ProgramOptionExists(argc, argv, "--register_sg"); // Keep it off. The function is still in developing.

    std::string corr_folder = result_folder + "/" + src_scene + "/" + ref_scene;

    // Global point cloud for visualization.
    // auto recons_src_pcd = io::CreatePointCloudFromFile(result_folder + "/" + src_scene + "/instance_map.ply");
    // auto recons_ref_pcd = io::CreatePointCloudFromFile(result_folder + "/" + ref_scene + "/instance_map.ply");
    // std::cout<<"Load reconstructed instance map from "<<result_folder<<std::endl;

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
    std::vector<float> corr_scores_vec;
    std::map<int, std::vector<int>> ins_corr_map;
    fmfusion::O3d_Cloud_Ptr corr_src_pcd(new fmfusion::O3d_Cloud()), corr_ref_pcd(new fmfusion::O3d_Cloud());

    int M;
    float ref_frame_timestamp;
    fmfusion::O3d_Cloud_Ptr src_pcd, ref_pcd;

    { // Load results
        bool load_results = fmfusion::IO::load_match_results(
                corr_folder + "/" + src_frame_name + ".txt",
                ref_frame_timestamp,
                pred_pose, match_pairs,
                src_centroids, ref_centroids,
                false);
        assert(load_results);
        assert(match_pairs.size() == src_centroids.size());
        M = match_pairs.size();
        std::cout << "Load match pairs: " << match_pairs.size() << std::endl;

        // Current dense map
        std::cout<<"Load src map from "<<corr_folder<<"/"<<src_frame_name<<"_src.ply"<<std::endl;
        src_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + src_frame_name + "_src.ply");
        ref_pcd = io::CreatePointCloudFromFile(ref_frame_map_dir);
        std::cout<<"load src pcd size: " << src_pcd->points_.size() << std::endl;
        std::cout << "load ref_pcd size: " << ref_pcd->points_.size() << std::endl;
        assert(ref_pcd->points_.size() > 0);
        if (!src_pcd->HasNormals()) src_pcd->EstimateNormals();
        if (!ref_pcd->HasNormals()) ref_pcd->EstimateNormals();
    }

    /*
    auto ref_map = std::make_shared<fmfusion::SemanticMapping>(
            fmfusion::SemanticMapping(sg_config->mapping_cfg, sg_config->instance_cfg));
    auto src_map = std::make_shared<fmfusion::SemanticMapping>(
            fmfusion::SemanticMapping(sg_config->mapping_cfg, sg_config->instance_cfg));

    std::cout << "Load ref map: " << output_folder + "/" + ref_scene << std::endl;
    ref_map->load(output_folder + "/" + ref_scene);
    src_map->load(output_folder + "/" + src_scene);
    ref_map->extract_bounding_boxes();
    src_map->extract_bounding_boxes();

    // Access instance points and bbox
    std::vector<fmfusion::InstancePtr> src_instances, ref_instances;
    for (int i = 0; i < match_pairs.size(); i++) {
        auto src_instance = src_map->get_instance(match_pairs[i].first);
        auto ref_instance = ref_map->get_instance(match_pairs[i].second);
        if (src_instance == nullptr && ref_instance == nullptr) {
            continue;
        }
        if (src_instance == nullptr) {
            src_instance = std::make_shared<fmfusion::Instance>(-1, -1, ref_instance->get_config());
            Eigen::Vector3d ref2src = src_centroids[i] - ref_centroids[i];
            for (int j = 0; j < ref_instance->point_cloud->points_.size(); j++) {
                src_instance->point_cloud->points_.emplace_back(ref_instance->point_cloud->points_[j] + ref2src);
            }
        }
        if (ref_instance == nullptr) {
            ref_instance = std::make_shared<fmfusion::Instance>(-1, -1, src_instance->get_config());
            Eigen::Vector3d src2ref = ref_centroids[i] - src_centroids[i];
            for (int j = 0; j < src_instance->point_cloud->points_.size(); j++) {
                ref_instance->point_cloud->points_.emplace_back(src_instance->point_cloud->points_[j] + src2ref);
            }
        }
        src_instances.push_back(src_instance);
        ref_instances.push_back(ref_instance);
    }
    if (!load_results) {
        utility::LogError("Failed to load match results.");
        return 0;
    }
    */

    // Dense matches

    // bool dense_mode = true;
    int dense_message = 0;
    corr_src_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + src_frame_name + "_csrc.ply");
    corr_ref_pcd = io::CreatePointCloudFromFile(corr_folder + "/" + src_frame_name + "_cref.ply");
    int C = corr_src_pcd->points_.size();
    assert(C == corr_ref_pcd->points_.size());
    std::cout << "Load " << corr_src_pcd->points_.size() << " dense corrs" << std::endl;

    // Dense message
    if (check_file_exists(corr_folder + "/" + src_frame_name + "_cmatches.txt")) {
        std::vector<int> corr_match_indces;
        fmfusion::IO::load_corrs_match_indices(corr_folder + "/" + src_frame_name + "_cmatches.txt",
                                               corr_match_indces,
                                               corr_scores_vec);
        assert(C == corr_scores_vec.size());

        if(downsample_corr){
            downsample_corr_nms(corr_src_pcd->points_, corr_ref_pcd->points_, corr_scores_vec, nms_thd);
            downsample_corr_topk(corr_src_pcd->points_, corr_ref_pcd->points_, corr_scores_vec, ds_voxel, max_corr_number);
        }
        dense_message = 1;

        // Demonstrate how to access the node index of each correpondence
        for (int i = 0; i < C; i++) {
            int match_index = corr_match_indces[i];
            assert(match_index < M);
            if (ins_corr_map.find(match_index) == ins_corr_map.end()) {
                ins_corr_map[match_index] = std::vector<int>();
            }
            ins_corr_map[match_index].push_back(i);
        }
    } else { // coarse message
        corr_scores_vec = std::vector<float>(C, 0.5);
        std::cout << "This is a coarse loop frame. Assign a constant corr score." << std::endl;
    }

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

    std::cout<<"icp_voxel: "<<icp_voxel
                <<"search_radius: "<<search_radius
                <<" ds_voxel: "<<ds_voxel<<std::endl;

    G3RegAPI g3reg(config);
    timer.Start();
    // g3reg.analyze_correspondences(src_centroids,
    //                               ref_centroids,
    //                               corr_src_pcd->points_,
    //                               corr_ref_pcd->points_,
    //                               ins_corr_map, gt_pose);

    if (corr_src_pcd->points_.size() > 0) {
        // dense registration
        double inlier_ratio = g3reg.estimate_pose_gnc(src_centroids, ref_centroids, corr_src_pcd->points_,
                                                      corr_ref_pcd->points_, corr_scores_vec);
        std::cout << "Inlier ratio: " << inlier_ratio << std::endl;
        if (inlier_ratio < inlier_threshold) {
            g3reg.estimate_pose(src_centroids, ref_centroids, 
                                corr_src_pcd->points_, corr_ref_pcd->points_,
                                corr_scores_vec, src_pcd, ref_pcd);
        }
    } else {
        assert(false);
    }

    std::vector<Eigen::Matrix4d> candidates = g3reg.reg_result.candidates;
    pred_pose = g3reg.reg_result.tf;
    timer.Stop();
    double g3reg_time = timer.GetDurationInMillisecond();

    timer.Start();
    if (enable_icp) {
        pred_pose = g3reg.icp_refine(src_pcd, ref_pcd, pred_pose);
        std::cout<<"ICP refine pose using "
                    << src_pcd->points_.size() << " src points and "
                    << ref_pcd->points_.size() << " ref points."<<std::endl;
    }
    timer.Stop();
    double icp_time = timer.GetDurationInMillisecond();

    std::cout << "Total time: " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() + g3reg_time
              << " ms, G3Reg time: " << std::fixed << std::setprecision(3) << g3reg_time << " ms, ICP time: "
              << std::fixed << std::setprecision(3) << icp_time << " ms\n";

    std::cout << "RMSE: " << eval_registration_error(src_pcd, pred_pose, gt_pose) << std::endl;

    if (visualization) {
        src_pcd->Transform(pred_pose);
        src_pcd->PaintUniformColor(Eigen::Vector3d(0.0, 0.651, 0.929));
        ref_pcd->PaintUniformColor(Eigen::Vector3d(1.0, 0.706, 0));
        visualization::DrawGeometries({src_pcd, ref_pcd}, "TestRegistration", 1920, 1080);
    }

    {
        std::cout << "Export results to " << new_result_folder << std::endl;
        fmfusion::IO::save_pose(new_result_folder + "/" + src_frame_name + "_newpose.txt", pred_pose);
        fmfusion::IO::write_time({"Msg, Clique, Graph, Solver, Verify, G3Reg, ICP"}, 
                                {dense_message,
                                g3reg.reg_result.clique_time,
                                g3reg.reg_result.graph_time,
                                g3reg.reg_result.tf_solver_time,
                                g3reg.reg_result.verify_time,
                                g3reg_time, icp_time}, 
                                new_result_folder +"/"+ src_frame_name + "_timing.txt");
        // io::WritePointCloudToPLY(export_folder + "/" + src_frame_name + "_src.ply", *src_pcd, {});
        // if (corr_src_pcd->points_.size() > 0 && corr_ref_pcd->points_.size() > 0) {
        //     io::WritePointCloud(export_folder + "/" + src_frame_name + "_csrc.ply", *corr_src_pcd);
        //     io::WritePointCloud(export_folder + "/" + src_frame_name + "_cref.ply", *corr_ref_pcd);
        // }
    }

    return 0;
}

