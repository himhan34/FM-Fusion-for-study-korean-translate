//
// Created by qzj on 6/24/24.
//
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/geometry/Pose3.h>

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
    std::vector<gtsam::Pose3> poses;
    poses.push_back(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 1.1, -0.9)));
    poses.push_back(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 0.9, -1.1)));
    poses.push_back(gtsam::Pose3(gtsam::Rot3(), gtsam::Point3(0, 1, -1)));
    Eigen::Matrix3d rot_rand = Eigen::Matrix3d::Random();
    Eigen::Vector3d trans_rand = Eigen::Vector3d::Random();
    poses.push_back(gtsam::Pose3(gtsam::Rot3(rot_rand), gtsam::Point3(trans_rand)));
    rot_rand = Eigen::Matrix3d::Random();
    trans_rand = Eigen::Vector3d::Random();
    poses.push_back(gtsam::Pose3(gtsam::Rot3(rot_rand), gtsam::Point3(trans_rand)));


    gtsam::Pose3 pose = gncRobustPoseAveraging(poses);
    std::cout << "pose: " << pose.matrix() << std::endl;
}