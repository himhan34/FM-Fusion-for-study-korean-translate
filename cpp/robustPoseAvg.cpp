//
// Created by qzj on 6/24/24.
//
#include "open3d/Open3D.h"

#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/geometry/Pose3.h>

#include "tools/IO.h"

gtsam::Pose3 gncRobustPoseAveraging(const std::vector<gtsam::Pose3> &input_poses,
                                    const double &rot_sigma = 0.1,
                                    const double &trans_sigma = 0.5) {
    gtsam::Values initial;
    initial.insert(0, gtsam::Pose3());  // identity pose as initialization

    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector sigmas;
    sigmas.resize(6);
    sigmas.head(3).setConstant(rot_sigma);
    sigmas.tail(3).setConstant(trans_sigma);
    const gtsam::noiseModel::Diagonal::shared_ptr noise =
            gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    // add measurements
    for (const auto &pose: input_poses) {
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose, noise));
    }

    gtsam::GncParams<gtsam::LevenbergMarquardtParams> gncParams;
    auto gnc = gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>>(graph, initial, gncParams);

    gnc.setInlierCostThresholdsAtProbability(0.99);

    gtsam::Values estimate = gnc.optimize();
    return estimate.at<gtsam::Pose3>(0);
}

int main(int argc, char **argv) {
    using namespace open3d;
    // SFM_DIR/SRC-REF or SGSLAM/output/v9/SRC/REF
    std::string input_folder = utility::GetProgramOptionAsString(argc, argv, "--input_folder"); 
    std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
    // std::string src_scene_folder = utility::GetProgramOptionAsString(argc, argv, "--src_scene_folder");
    // std::string ref_scene_folder = utility::GetProgramOptionAsString(argc, argv, "--ref_scene_folder");
    std::string anchor_frame = utility::GetProgramOptionAsString(argc, argv, "--anchor_frame");
    std::string frame_list = utility::GetProgramOptionAsString(argc, argv, "--frame_list", ""); // frame-xxxxxx frame-xxxxxx ...
    bool sfm = utility::ProgramOptionExists(argc, argv, "--sfm");

    // Prepare frame names
    int num_frames = (frame_list.size() / 12);
    std::cout<< num_frames<<" frames"<<std::endl;
    std::vector<std::string> target_frame_names = {anchor_frame};

    for (int k=0;k<num_frames;k++){
        std::string frame_name = frame_list.substr(k*12, 12);
        // std::cout<<frame_name<<std::endl;
        if (frame_name != anchor_frame){
            target_frame_names.push_back(frame_name);
        }
    }

    // Load src and ref poses
    std::unordered_map<std::string, Eigen::Matrix4d> src_poses_map;
    std::unordered_map<std::string, Eigen::Matrix4d> ref_poses_map;
    std::vector<Eigen::Matrix4d> pose_table;

    if (sfm){
        fmfusion::IO::read_frames_poses(input_folder +"/pose_graph/src_poses.txt", src_poses_map);
        fmfusion::IO::read_frames_poses(input_folder +"/pose_graph/ref_poses.txt", ref_poses_map);

        // Load T_ref_src measurements
        for (const auto &frame_name: target_frame_names){
            std::vector<fmfusion::LoopPair> loop_pairs;
            std::vector<Eigen::Matrix4d> loop_transformations;
            fmfusion::IO::read_loop_transformations(input_folder+"/pnp/"+frame_name+".txt", 
                                                    loop_pairs, loop_transformations);
            if(ref_poses_map.find(loop_pairs[0].second) == ref_poses_map.end() ||
            src_poses_map.find(loop_pairs[0].first) == src_poses_map.end()){
                std::cout<<"No PnP transformation found for "<<loop_pairs[0].first<<" -> "<<loop_pairs[0].second<<std::endl;
                continue;
            }

            Eigen::Matrix4d T_c1_c0 = loop_transformations[0];
            Eigen::Matrix4d T_ref_c1 = ref_poses_map[loop_pairs[0].second];
            Eigen::Matrix4d T_src_c0 = src_poses_map[loop_pairs[0].first];
            Eigen::Matrix4d T_ref_src = T_ref_c1 * T_c1_c0 * T_src_c0.inverse();

            // std::cout<<"T_c1_c0\n"<<T_c1_c0<<std::endl;
            // std::cout<<"T_ref_c1\n"<<T_ref_c1<<std::endl;
            // std::cout<<"T_src_c0\n"<<T_src_c0<<std::endl;
            std::cout<<"T_ref_src\n"<<T_ref_src<<std::endl;

            pose_table.push_back(T_ref_src);
        }
        
    }
    else{
        // fmfusion::IO::read_entire_camera_poses(src_scene_folder, src_poses_map);
        // fmfusion::IO::read_entire_camera_poses(ref_scene_folder, ref_poses_map);
        
        // Load T_ref_src measurements
        for(const auto &frame_name: target_frame_names){
            float ref_timestamp;
            Eigen::Matrix4d T_ref_src;
            std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
            std::vector<Eigen::Vector3d> src_centroids;
            std::vector<Eigen::Vector3d> ref_centroids;

            fmfusion::IO::load_match_results(input_folder+"/"+frame_name+".txt",
                                            ref_timestamp, 
                                            T_ref_src, 
                                            match_pairs, 
                                            src_centroids, ref_centroids);
            pose_table.push_back(T_ref_src);
        }

    }

    std::cout<<"Load "<< pose_table.size()<<" poses"<<std::endl;

    if(pose_table.empty()){
        std::cout<<"No poses to optimize"<<std::endl;
        return 0;
    }

    // Pose Average
    std::vector<gtsam::Pose3> poses;
    for (const auto &pose: pose_table){
        Eigen::Matrix3d rot = pose.block<3,3>(0,0);
        Eigen::Vector3d pos = pose.block<3,1>(0,3);
        poses.push_back(gtsam::Pose3(gtsam::Rot3(rot), gtsam::Point3(pos)));
    }

    // poses.push_back(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 1.1, -0.9)));
    // poses.push_back(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0.9, -1.1)));
    // poses.push_back(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 1, -1)));
    // Eigen::Matrix3d rot_rand = Eigen::Matrix3d::Random();
    // Eigen::Vector3d trans_rand = Eigen::Vector3d::Random();
    // poses.push_back(gtsam::Pose3(gtsam::Rot3(rot_rand), gtsam::Point3(trans_rand)));
    // rot_rand = Eigen::Matrix3d::Random();
    // trans_rand = Eigen::Vector3d::Random();
    // poses.push_back(gtsam::Pose3(gtsam::Rot3(rot_rand), gtsam::Point3(trans_rand)));

    gtsam::Pose3 pose = gncRobustPoseAveraging(poses);
    std::cout<<"original T_ref_src : \n"<<pose_table[0]<<std::endl;
    std::cout << "optimized T_ref_src: \n" << pose.matrix() << std::endl;

    Eigen::Matrix4d output_pose = pose.matrix();

    fmfusion::IO::save_pose(output_folder+"/"+anchor_frame+".txt", 
                            output_pose);

}