//
// Created by qzj on 6/24/24.
//
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/geometry/Pose3.h>

namespace registration
{
gtsam::Pose3 gncRobustPoseAveraging(const std::vector<gtsam::Pose3> &input_poses,
                                    const double &rot_sigma = 0.1,
                                    const double &trans_sigma = 0.5);
} // namespace registration