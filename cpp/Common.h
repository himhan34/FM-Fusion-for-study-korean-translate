#ifndef FMFUSOIN_COMMON_H
#define FMFUSOIN_COMMON_H

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "cluster/PoseGraph.h"

namespace fmfusion
{
using namespace std;

struct Config
{
    enum DATASET_TYPE{
        REALSENSE,
        FUSION_PORTABLE,
        SCANNET,
        MATTERPORT,
        RIO
    }dataset;
    
    // camera
    open3d::camera::PinholeCameraIntrinsic intrinsic;
    double depth_scale;
    double depth_max;

    // volumetric
    double voxel_length;
    double sdf_trunc;
    int min_active_points;    

    // associations
    int min_det_masks;
    double max_box_area_ratio;
    double query_depth_vx_size; // <0 if skipped
    double search_radius;
    int dilation_size;
    double min_iou;

    // shape
    // double cluster_eps;
    // int cluster_min_points;
    double min_voxel_weight;
    int shape_min_points=1000;

    // merge
    double merge_iou;
    double merge_inflation;

    // output
    int cleanup_period=20; // in frames
    bool save_da_images;
    std::string tmp_dir; // For debug
};

typedef std::shared_ptr<cv::Mat> CvMatPtr;
typedef open3d::geometry::Image O3d_Image;
typedef std::shared_ptr<open3d::geometry::Image> O3d_Image_Ptr;

}

#endif //FMFUSOIN_COMMON_H