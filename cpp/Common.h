#ifndef FMFUSOIN_COMMON_H
#define FMFUSOIN_COMMON_H

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"


namespace fmfusion
{
using namespace std;

struct Config
{
    enum DATASET_TYPE{
        REALSENSE,
        FUSION_PORTABLE,
        SCANNET
    }dataset;
    
    // camera
    open3d::camera::PinholeCameraIntrinsic intrinsic;
    double depth_scale;
    double depth_max;

    // volumetric
    double voxel_length;
    double sdf_trunc;
    int min_instance_points;

    // associations
    int min_det_masks;
    double max_box_area_ratio;
    int min_instance_masks;
    int dilation_size;
    double min_iou;

    // output
    bool save_da_images;
    std::string tmp_dir; // For debug
};

namespace o3d_utility = open3d::utility;
typedef std::shared_ptr<cv::Mat> CvMatPtr;
typedef open3d::geometry::Image O3d_Image;
typedef open3d::geometry::PointCloud O3d_Cloud;
typedef std::shared_ptr<open3d::geometry::PointCloud> O3d_Cloud_Ptr;

typedef uint32_t InstanceId;

}

#endif //FMFUSOIN_COMMON_H