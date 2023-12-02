#ifndef FMFUSION_UTILITY_H
#define FMFUSION_UTILITY_H

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "Common.h"
#include "Detection.h"

namespace fmfusion
{

namespace utility
{

std::vector<std::string> 
    split_str(const std::string s, const std::string delim);

fmfusion::Config *create_scene_graph_config(const std::string &config_file, bool verbose);

std::string config_to_message(const fmfusion::Config &config);

bool LoadPredictions(const std::string &folder_path, const std::string &frame_name, const Config &config, std::vector<DetectionPtr> &detections);

std::shared_ptr<cv::Mat> RenderDetections(const std::shared_ptr<cv::Mat> &rgb_img,
    const std::vector<fmfusion::DetectionPtr> &detections, const std::unordered_map<InstanceId,CvMatPtr> &instances_mask,
    const Eigen::VectorXi &matches, const std::unordered_map<InstanceId,Eigen::Vector3d> &instance_colors);

std::shared_ptr<cv::Mat> PrjectionCloudToDepth(const open3d::geometry::PointCloud& cloud, 
    const Eigen::Matrix4d &pose_inverse,const open3d::camera::PinholeCameraIntrinsic& intrinsic, int dilation_size);

bool create_masked_rgbd(
    const open3d::geometry::Image &rgb, const open3d::geometry::Image &float_depth, const cv::Mat &mask,
    const int &min_points,
    std::shared_ptr<open3d::geometry::RGBDImage> &masked_rgbd);
}

O3d_Image_Ptr extract_masked_o3d_image(const O3d_Image &depth, const O3d_Image &mask);

}

#endif //FMFUSION_UTILITY_H