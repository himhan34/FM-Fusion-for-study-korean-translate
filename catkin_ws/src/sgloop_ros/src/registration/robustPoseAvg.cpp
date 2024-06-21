#include "registration/robustPoseAvg.h"

namespace registration
{
gtsam::Pose3 gncRobustPoseAveraging(const std::vector<gtsam::Pose3> &input_poses,
                                    const double &rot_sigma,
                                    const double &trans_sigma) {
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


} // namespace registration