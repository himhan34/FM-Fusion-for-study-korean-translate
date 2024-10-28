//
// Created by qzj on 6/27/24.
//

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include "open3d/Open3D.h"
#include "tools/IO.h"
#include "tools/TicToc.h"

#include "KimeraRPGO/SolverParams.h"
#include "KimeraRPGO/outlier/Pcm.h"

using KimeraRPGO::OutlierRemoval;
using KimeraRPGO::Pcm3D;
using KimeraRPGO::PcmParams;

std::string zero_padding(const int &num, const int &num_digits)
{
    std::string num_str = std::to_string(num);
    int num_zeros = num_digits - num_str.size();
    std::string zero_str = "";
    for(int i=0;i<num_zeros;i++){
        zero_str += "0";
    }
    return zero_str + num_str;
}

void read_poses(const std::string &filename,
                std::vector<Eigen::Isometry3d> &poses,
                std::map<std::string, int> &frame_id_map) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    // ignore the first line
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string frame_id;
        double x, y, z, qx, qy, qz, qw;
        iss >> frame_id >> x >> y >> z >> qx >> qy >> qz >> qw;
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(x, y, z);
        pose.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
        poses.push_back(pose);
        frame_id_map[frame_id] = poses.size() - 1;
    }
    file.close();
}

bool read_loop(
        const std::string &filename,
        std::vector<std::tuple<std::string, std::string, Eigen::Isometry3d>>
        &loops) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    std::string line;
    // ignore the first line
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string src_frame, ref_frame;
        double x, y, z, qx, qy, qz, qw;
        iss >> src_frame >> ref_frame >> x >> y >> z >> qx >> qy >> qz >> qw;
        Eigen::Isometry3d loop = Eigen::Isometry3d::Identity();
        loop.translation() = Eigen::Vector3d(x, y, z);
        loop.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
        loops.push_back(std::make_tuple(src_frame, ref_frame, loop));
    }
    file.close();
    return true;
}

void read_pose(const std::string &filename, Eigen::Isometry3d &pose) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    Eigen::Matrix4d T;
    file >> T(0, 0) >> T(0, 1) >> T(0, 2) >> T(0, 3);
    file >> T(1, 0) >> T(1, 1) >> T(1, 2) >> T(1, 3);
    file >> T(2, 0) >> T(2, 1) >> T(2, 2) >> T(2, 3);
    file >> T(3, 0) >> T(3, 1) >> T(3, 2) >> T(3, 3);
    pose.matrix() = T;
    file.close();
}

bool save_pose(const std::string &filename,
                const std::string &src_frame_name,
                const std::string &ref_frame_name,
                const Eigen::Matrix4d &T)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    file<<"# src_frame ref_frame x y z qx qy qz qw"<<std::endl;
    file << src_frame_name << " " << ref_frame_name << " ";
    file <<std::fixed<<std::setprecision(3) 
        << T(0, 3) << " " << T(1, 3) << " " << T(2, 3) << " ";
    
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    file<< std::setprecision(6) 
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

    file.close();

    std::cout<<"Save pose to "<<filename<<std::endl;

}
                
void register_multi_session(std::string scene_folder, int frame_id) {

    // Read src and ref poses
    std::string src_poses_file = scene_folder + "/pose_graph/src_poses.txt";
    std::string ref_poses_file = scene_folder + "/pose_graph/ref_poses.txt";
    std::vector<Eigen::Isometry3d> src_poses, ref_poses;
    std::map<std::string, int> src_frame_id_map, ref_frame_id_map;
    read_poses(src_poses_file, src_poses, src_frame_id_map);
    read_poses(ref_poses_file, ref_poses, ref_frame_id_map);
    std::cout<<"Load "<<src_frame_id_map.size()<<" src poses from "<<src_poses_file<<std::endl;

    // Read true frame transformation
    std::string true_frame_tf_file = scene_folder + "/pose_graph" + "/gt.txt";
    Eigen::Isometry3d T_ref_src = Eigen::Isometry3d::Identity();
    read_pose(true_frame_tf_file, T_ref_src);

    // Read loop transformations
    std::string loop_file = scene_folder + "/loop_transformations.txt";
    std::vector<std::tuple<std::string, std::string, Eigen::Isometry3d>> loops;
    read_loop(loop_file, loops);
    std::vector<std::tuple<int, int, Eigen::Isometry3d>> loop_edges;
    for (const auto &loop: loops) {
        const std::string &src_frame = std::get<0>(loop);
        const std::string &ref_frame = std::get<1>(loop);
        const Eigen::Isometry3d &loop_transformation = std::get<2>(loop);
        if (src_frame_id_map.find(src_frame) == src_frame_id_map.end() ||
            ref_frame_id_map.find(ref_frame) == ref_frame_id_map.end()) {
            std::cerr << "Frame not found: " << src_frame << " " << ref_frame
                      << std::endl;
            continue;
        }
        int src_frame_id = src_frame_id_map[src_frame];
        int ref_frame_id = ref_frame_id_map[ref_frame];
        std::cout<<"<"<<src_frame_id<<","<<ref_frame_id<<">,";
        loop_edges.push_back(
                std::make_tuple(src_frame_id, ref_frame_id, loop_transformation));
    }
    std::cout<<std::endl;

    // Create PCM
    PcmParams params;
    params.odom_trans_threshold = -1;
    params.odom_rot_threshold = -1;
    params.dist_trans_threshold = 0.5;
    params.dist_rot_threshold = 100.0;

    // Create PCM
    OutlierRemoval *pcm =
            new Pcm3D(params, KimeraRPGO::MultiRobotAlignMethod::GNC);
    // pcm->setQuiet();

    static const gtsam::SharedNoiseModel &ego_noise =
            gtsam::noiseModel::Isotropic::Variance(6, 0.1);
    static const gtsam::SharedNoiseModel &prior_noise =
            gtsam::noiseModel::Isotropic::Variance(6, 0.0001);
    static const gtsam::SharedNoiseModel &loop_noise =
            gtsam::noiseModel::Isotropic::Variance(6, 0.1);

    gtsam::NonlinearFactorGraph nfg;
    gtsam::Values est;

    // Add src robot
    for (int i = 0; i < src_poses.size(); ++i) {
        est.insert(gtsam::Symbol('a', i), gtsam::Pose3(src_poses[i].matrix()));
    }
    for (int i = 0; i < src_poses.size() - 1; ++i) {
        Eigen::Isometry3d ego_transformation =
                src_poses[i].inverse() * src_poses[i + 1];
        nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('a', i),
                gtsam::Symbol('a', i + 1),
                gtsam::Pose3(ego_transformation.matrix()),
                ego_noise));
    }

    // Add prior for the first src pose
    // nfg.add(gtsam::PriorFactor<gtsam::Pose3>(
    //    gtsam::Symbol('a', 0), gtsam::Pose3(src_poses[0].matrix()),
    //    prior_noise));

    // Add ref robot
    for (int i = 0; i < ref_poses.size(); ++i) {
        est.insert(gtsam::Symbol('b', i), gtsam::Pose3(ref_poses[i].matrix()));
    }
    for (int i = 0; i < ref_poses.size() - 1; ++i) {
        Eigen::Isometry3d ego_transformation =
                ref_poses[i].inverse() * ref_poses[i + 1];
        nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('b', i),
                gtsam::Symbol('b', i + 1),
                gtsam::Pose3(ego_transformation.matrix()),
                ego_noise));
    }

    // Add loop closures
    for (const auto &loop_edge: loop_edges) {
        int src_frame_id = std::get<0>(loop_edge);
        int ref_frame_id = std::get<1>(loop_edge);
        std::cout<<"<"<<src_frame_id<<","<<ref_frame_id<<">,";
        const Eigen::Isometry3d &loop_transformation = std::get<2>(loop_edge);
        nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('a', src_frame_id),
                gtsam::Symbol('b', ref_frame_id),
                gtsam::Pose3(loop_transformation.inverse().matrix()),
                loop_noise));
    }
    std::cout<<std::endl;

    gtsam::NonlinearFactorGraph output_nfg;
    gtsam::Values output_vals;
    pcm->removeOutliers(nfg, est, &output_nfg, &output_vals);

    std::cout << "GT T_ref_src: " << std::endl;
    std::cout << "Euler angle: " << T_ref_src.rotation().eulerAngles(2, 1, 0).transpose() * 180.0 / M_PI << std::endl;
    std::cout << "Translation: " << T_ref_src.translation().transpose() << std::endl;
    Eigen::Matrix4d T_ref_src_est =
            ref_poses[0] *
            output_vals.at<gtsam::Pose3>(gtsam::Symbol('b', 0)).matrix().inverse();
    std::cout << "Est T_ref_src: " << std::endl;
    std::cout << "Euler angle: " << T_ref_src_est.block<3, 3>(0, 0).eulerAngles(2, 1, 0).transpose() * 180.0 / M_PI
              << std::endl;
    std::cout << "Translation: " << T_ref_src_est.block<3, 1>(0, 3).transpose() << std::endl;

    // Save
    std::string last_src_frame = std::get<0>(loops.back());
    std::string last_ref_frame = std::get<1>(loops.back());
    std::cout<<"Save pose: "<<last_src_frame<<" "<<last_ref_frame<<std::endl;

    save_pose(scene_folder + "/pose_graph/summary_pgo.txt", 
            last_src_frame,
            last_ref_frame,
            T_ref_src_est);


}

void register_multi_agent(std::string scene_folder, int frame_id, int window_size)
{
    // Read loop transformations in a window
    std::vector<std::tuple<std::string, std::string, Eigen::Isometry3d>> loops;
    std::vector<std::string> src_window_frames;
    int idx=frame_id;
    while(idx>=0 && idx>=(frame_id-window_size)){
        std::string frame_name = "frame-"+ zero_padding(idx, 6);
        std::string loop_file = scene_folder + "/pnp/"+ frame_name +".txt";
        if(read_loop(loop_file, loops)){
            src_window_frames.push_back(frame_name);
            std::cout<<"Load "<<loops.size()<<" loops from  "<<frame_name<<std::endl;
        }
        idx -= 10;
    }
    if(loops.empty()) return;

    // Read src and ref poses
    std::string src_poses_file = scene_folder + "/pose_graph/src_poses.txt";
    std::string ref_poses_file = scene_folder + "/pose_graph/ref_poses.txt";
    std::vector<Eigen::Isometry3d> src_poses, ref_poses;
    std::map<std::string, int> src_frame_id_map, ref_frame_id_map;
    read_poses(src_poses_file, src_poses, src_frame_id_map);
    read_poses(ref_poses_file, ref_poses, ref_frame_id_map);
    std::cout<<"Load "<<src_frame_id_map.size()<<" src poses from "<<src_poses_file<<std::endl;

    // Construct loop edges
    std::vector<std::tuple<int, int, Eigen::Isometry3d>> loop_edges;
    for (const auto &loop: loops) {
        const std::string &src_frame = std::get<0>(loop);
        const std::string &ref_frame = std::get<1>(loop);
        const Eigen::Isometry3d &loop_transformation = std::get<2>(loop);
        if (src_frame_id_map.find(src_frame) == src_frame_id_map.end() ||
            ref_frame_id_map.find(ref_frame) == ref_frame_id_map.end()) {
            std::cerr << "Frame not found: " << src_frame << " " << ref_frame
                      << std::endl;
            continue;
        }
        int src_frame_id = src_frame_id_map[src_frame];
        int ref_frame_id = ref_frame_id_map[ref_frame];
        std::cout<<"<"<<src_frame_id<<","<<ref_frame_id<<">,";
        loop_edges.push_back(
                std::make_tuple(src_frame_id, ref_frame_id, loop_transformation));
    }
    std::cout<<std::endl;

    // Create PCM
    PcmParams params;
    params.odom_trans_threshold = -1;
    params.odom_rot_threshold = -1;
    params.dist_trans_threshold = 0.5;
    params.dist_rot_threshold = 100.0;

    // Create PCM
    OutlierRemoval *pcm =
            new Pcm3D(params, KimeraRPGO::MultiRobotAlignMethod::GNC);
    // pcm->setQuiet();

    static const gtsam::SharedNoiseModel &ego_noise =
            gtsam::noiseModel::Isotropic::Variance(6, 0.1);
    static const gtsam::SharedNoiseModel &prior_noise =
            gtsam::noiseModel::Isotropic::Variance(6, 0.0001);
    static const gtsam::SharedNoiseModel &loop_noise =
            gtsam::noiseModel::Isotropic::Variance(6, 0.1);

    gtsam::NonlinearFactorGraph nfg;
    gtsam::Values est;

    // Add src odometry constraints
    // for(const auto &frame_name: src_window_frames){
    //     assert(src_frame_id_map.find(frame_name) != src_frame_id_map.end());
    //     int i = src_frame_id_map[frame_name];
    for (int i=0; i<src_poses.size(); ++i) {
        est.insert(gtsam::Symbol('a', i), gtsam::Pose3(src_poses[i].matrix()));
    }
    // for(const auto &frame_name: src_window_frames){
    //     int i = src_frame_id_map[frame_name];
    for (int i=0; i<src_poses.size()-1; ++i) {
        Eigen::Isometry3d ego_transformation =
                src_poses[i].inverse() * src_poses[i + 1];
        nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('a', i),
                gtsam::Symbol('a', i + 1),
                gtsam::Pose3(ego_transformation.matrix()),ego_noise));
    }

    // Add ref robot
    for (int i = 0; i < ref_poses.size(); ++i) {
        est.insert(gtsam::Symbol('b', i), gtsam::Pose3(ref_poses[i].matrix()));
    }
    for (int i = 0; i < ref_poses.size() - 1; ++i) {
        Eigen::Isometry3d ego_transformation =
                ref_poses[i].inverse() * ref_poses[i + 1];
        nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('b', i),
                gtsam::Symbol('b', i + 1),
                gtsam::Pose3(ego_transformation.matrix()),ego_noise));
    }

    // Add loop closures
    for (const auto &loop_edge: loop_edges) {
        int src_frame_id = std::get<0>(loop_edge);
        int ref_frame_id = std::get<1>(loop_edge);
        const Eigen::Isometry3d &loop_transformation = std::get<2>(loop_edge);
        std::cout<<"<"<<src_frame_id<<","<<ref_frame_id<<">,";
        nfg.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol('a', src_frame_id),
                gtsam::Symbol('b', ref_frame_id),
                gtsam::Pose3(loop_transformation.inverse().matrix()),
                loop_noise));
    }
    std::cout<<std::endl;

    // Optimization
    std::cout<<"Optimization..."<<std::endl;
    gtsam::NonlinearFactorGraph output_nfg;
    gtsam::Values output_vals;
    pcm->removeOutliers(nfg, est, &output_nfg, &output_vals);

    // Print results
    Eigen::Matrix4d T_ref_src_est =
            ref_poses[0] *
            output_vals.at<gtsam::Pose3>(gtsam::Symbol('b', 0)).matrix().inverse();
    std::cout << "Est T_ref_src: " << std::endl;
    std::cout << "Euler angle: " << T_ref_src_est.block<3, 3>(0, 0).eulerAngles(2, 1, 0).transpose() * 180.0 / M_PI
              << std::endl;
    std::cout << "Translation: " << T_ref_src_est.block<3, 1>(0, 3).transpose() << std::endl;

    // Save results
    fmfusion::IO::save_pose(scene_folder + "/pgo/frame-"+ zero_padding(frame_id, 6) + ".txt", 
            T_ref_src_est);
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    std::string scene_folder = utility::GetProgramOptionAsString(argc, argv, "--scene_folder");
    // std::string frame_name = utility::GetProgramOptionAsString(argc, argv, "--frame_name");
    int frame_id = utility::GetProgramOptionAsInt(argc, argv, "--frame_id", 0);
    int window_size = utility::GetProgramOptionAsInt(argc, argv, "--window_size", 50);
    int mode = utility::GetProgramOptionAsInt(argc, argv, "--mode", 1);

    if(mode==0)
        register_multi_session(scene_folder, frame_id);
    else
        register_multi_agent(scene_folder, frame_id, window_size);

    return 0;
}