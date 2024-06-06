#ifndef FMFUSOIN_COMMON_H
#define FMFUSOIN_COMMON_H

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "cluster/PoseGraph.h"
// #include "sgloop/Graph.h"

namespace fmfusion
{
using namespace std;

struct GraphConfig{
    float edge_radius_ratio = 2.0;
    float voxel_size = 0.02;
    bool involve_floor_edge = false;
    std::string ignore_labels = "floor. carpet. ceiling.";

    const std::string print_msg()const{
        std::stringstream msg;
        msg<<" - edge_radius_ratio: "<<edge_radius_ratio<<std::endl;
        msg<<" - voxel_size: "<<voxel_size<<std::endl;
        msg<<" - involve_floor_edge: "<<involve_floor_edge<<std::endl;
        msg<<" - ignore_labels: "<<ignore_labels<<std::endl;
        return msg.str();
    }
};

struct SgNetConfig
{
    int token_padding=8;
    int triplet_number=20; // number of triplets for each node
    float instance_match_threshold=0.1;

    const std::string print_msg()const{
        std::stringstream msg;
        msg<<" - token_padding: "<<token_padding<<std::endl;
        msg<<" - triplet_number: "<<triplet_number<<std::endl;
        msg<<" - instance_match_threshold: "<<instance_match_threshold<<std::endl;
        return msg.str();
    }
};


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

    //
    GraphConfig graph;
    SgNetConfig sgnet;
        
};

typedef std::shared_ptr<cv::Mat> CvMatPtr;
typedef open3d::geometry::Image O3d_Image;
typedef open3d::geometry::Geometry O3d_Geometry;
typedef std::shared_ptr<const open3d::geometry::Geometry> O3d_Geometry_Ptr;
typedef std::shared_ptr<open3d::geometry::Image> O3d_Image_Ptr;

}

#endif //FMFUSOIN_COMMON_H