#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "Utility.h"

namespace fmfusion
{

namespace utility
{

std::vector<std::string> split_str(const std::string s, const std::string delim) 
{
    std::vector<std::string> list;
    auto start = 0U;
    auto end = s.find(delim);
    while (true) {
        list.push_back(s.substr(start, end - start));
        if (end == std::string::npos)
            break;
        start = end + delim.length();
        end = s.find(delim, start);
    }
    return list;
}

fmfusion::Config *create_scene_graph_config(const std::string &config_file, bool verbose)
{
    fmfusion::Config *config = new fmfusion::Config();
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        open3d::utility::LogWarning("Failed to open config file: {}", config_file);
        return nullptr;
    }
    else{
        std::string dataset_name = fs["dataset"];
        if(dataset_name.find("fusion_portable")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::FUSION_PORTABLE;
        else if(dataset_name.find("realsense")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::REALSENSE;
        else if(dataset_name.find("scannet")!=string::npos)
            config->dataset = fmfusion::Config::DATASET_TYPE::SCANNET;
        
        int img_width = fs["image_width"];
        int img_height = fs["image_height"];
        double fx = fs["camera_fx"];
        double fy = fs["camera_fy"];
        double cx = fs["camera_cx"];
        double cy = fs["camera_cy"];

        config->intrinsic.SetIntrinsics(img_width,img_height,fx,fy,cx,cy);
        config->depth_scale = fs["depth_scale"];
        config->depth_max = fs["depth_max"];

        config->voxel_length = fs["voxel_length"];
        config->sdf_trunc = fs["sdf_trunc"];
        config->min_instance_points = fs["min_instance_points"];

        config->min_det_masks = fs["min_det_masks"];
        config->max_box_area_ratio = fs["max_box_area_ratio"];
        // config->min_instance_masks = fs["min_instance_masks"];
        config->dilation_size = fs["dilate_kernal"];
        config->min_iou = fs["min_iou"];

        // config->cluster_eps = fs["cluster_eps"];
        // config->cluster_min_points = fs["cluster_min_points"];
        config->min_voxel_weight = fs["min_voxel_weight"];

        config->merge_iou = fs["merge_iou"];
        config->merge_inflation = fs["merge_inflation"];

        int save_da_images = fs["save_da_images"];
        if (save_da_images>0) config->save_da_images = true;
        else config->save_da_images = false;
        
        // fs["tmp_dir"]>>config->tmp_dir;

        // Close and print
        fs.release();
        if (verbose){
            auto message = config_to_message(*config);
            cout << message << endl;
        }        
        return config;
    }
}

std::string config_to_message(const fmfusion::Config &config)
{
    std::stringstream message;
    switch (config.dataset)
    {
    case fmfusion::Config::DATASET_TYPE::FUSION_PORTABLE:
        message << "dataset: fusion_portable\n";
        break;
    case fmfusion::Config::DATASET_TYPE::REALSENSE:
        message << "dataset: realsense\n";
        break;
    case fmfusion::Config::DATASET_TYPE::SCANNET:
        message << "dataset: scannet\n";
        break;
    default:
        break;
    }
    
    message << "image_width: " + std::to_string(config.intrinsic.width_) + "\n";
    message << "image_height: " + std::to_string(config.intrinsic.height_) + "\n";
    message << "camera_fx: " + std::to_string(config.intrinsic.intrinsic_matrix_(0,0)) + "\n";
    message << "camera_fy: " + std::to_string(config.intrinsic.intrinsic_matrix_(1,1)) + "\n";
    message << "camera_cx: " + std::to_string(config.intrinsic.intrinsic_matrix_(0,2)) + "\n";
    message << "camera_cy: " + std::to_string(config.intrinsic.intrinsic_matrix_(1,2)) + "\n";

    message << "voxel_length: " << config.voxel_length << "\n";
    message << "sdf_trunc: " << std::to_string(config.sdf_trunc) + "\n";
    message << "min_instance_points: " + std::to_string(config.min_instance_points) + "\n";

    message << "min_det_masks: " + std::to_string(config.min_det_masks) + "\n";
    message << "max_box_area_ratio: "<< std::fixed<<std::setprecision(2)<<config.max_box_area_ratio << "\n";
    message << "dilate_kernal: " + std::to_string(config.dilation_size) + "\n";
    message << "min_iou: "<< std::fixed<<std::setprecision(2)<<config.min_iou << "\n";

    message << "min_voxel_weight: " << std::fixed<<std::setprecision(2)<<config.min_voxel_weight << "\n";

    message << "save_da_images: " + std::to_string(config.save_da_images) + "\n";
    // message << "tmp_dir: " + config.tmp_dir + "\n";

    return message.str();
}



bool LoadPredictions(const std::string &folder_path, const std::string &frame_name, const Config &config,
    std::vector<DetectionPtr> &detections)
{
    const int MAX_BOX_AREA = config.max_box_area_ratio * (config.intrinsic.width_ * config.intrinsic.height_);

    auto detection_fs = std::make_shared<fmfusion::DetectionFile>(config.min_det_masks,MAX_BOX_AREA);
    std::string json_file_dir = folder_path + "/" + frame_name + "_label.json";
    std::string instance_file_dir = folder_path + "/" + frame_name + "_mask.png";

    if(open3d::io::ReadIJsonConvertible(json_file_dir, *detection_fs)){
        bool read_mask = detection_fs->updateInstanceMap(instance_file_dir);
        detections = detection_fs->detections;
        cout<<"Load "<<detections.size()<<" detections correct"<<endl;
        return true;
    }
    else return false;

}



std::shared_ptr<cv::Mat> RenderDetections(const std::shared_ptr<cv::Mat> &rgb_img,
    const std::vector<DetectionPtr> &detections, const std::unordered_map<InstanceId,CvMatPtr> &instances_mask,
    const Eigen::VectorXi &matches, const std::unordered_map<InstanceId,Eigen::Vector3d> &instance_colors)
{
    auto detection_img = std::make_shared<cv::Mat>(rgb_img->clone());
    auto detection_mask = std::make_shared<cv::Mat>(cv::Mat::zeros(rgb_img->rows, rgb_img->cols, CV_8UC3));
    auto instance_img = std::make_shared<cv::Mat>(cv::Mat::zeros(rgb_img->rows, rgb_img->cols, CV_8UC3));
    if(detections.size()<1) return detection_img;

    int k=0;
    for(auto detection:detections){
        cv::Scalar box_color;
        if(matches(k)>0) box_color = cv::Scalar(0,255,0);
        else if(matches(k)<0) box_color = cv::Scalar(0,0,255);  // invalid
        else box_color = cv::Scalar(255,0,0); // create new

        cv::rectangle(*detection_img, cv::Point(detection->bbox_.u0,detection->bbox_.v0),
            cv::Point(detection->bbox_.u1,detection->bbox_.v1), box_color, 1);
        std::string label_score_str = detection->extract_label_string();
        cv::putText(*detection_img, label_score_str, cv::Point(detection->bbox_.u0+1,detection->bbox_.v0+10), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
        k++;

        // detection-wise color
        cv::Scalar det_color = cv::Scalar(rand()%255,rand()%255,rand()%255);
        detection_mask->setTo(det_color, detection->instances_idxs_);
    }

    for (auto &instance:instances_mask){
        cv::Scalar inst_color_cv;
        if(instance_colors.empty())
            inst_color_cv = cv::Scalar(rand()%255,rand()%255,rand()%255);
        else{
            const Eigen::Vector3d inst_color_vec = 255 * instance_colors.at(instance.first);
            inst_color_cv = cv::Scalar(inst_color_vec[0],inst_color_vec[1],inst_color_vec[2]);
        }

        instance_img->setTo(inst_color_cv, *instance.second);
    }

    // concatenate images
    cv::addWeighted(*detection_img, 1.0, *detection_mask, 0.5, 0.0, *detection_img);
    auto out_img = std::make_shared<cv::Mat>(cv::Mat::zeros(rgb_img->rows, rgb_img->cols*2, CV_8UC3));
    cv::hconcat(*detection_img, *instance_img, *out_img);

    // draw matches
    int K = matches.size();
    cv::Scalar line_color = cv::Scalar(255,255,0);
    for(int k_=0;k_<K;k_++){
        if(matches(k_)>0){
            cv::Point detection_centroid = detections[k_]->get_box_center();
            auto matched_instance = instances_mask.find(matches(k_)); // [H,W], CV_8UC1
            cv::Mat instance_uvs;
            cv::findNonZero(*matched_instance->second, instance_uvs);
            cv::Point instance_centroid = cv::Point(cv::mean(instance_uvs)[0],cv::mean(instance_uvs)[1]);
            cv::line(*out_img, detection_centroid, cv::Point(instance_centroid.x+rgb_img->cols,instance_centroid.y), line_color, 1);
        }
    }


    return out_img;
}


std::shared_ptr<cv::Mat> PrjectionCloudToDepth(const open3d::geometry::PointCloud& cloud, 
    const Eigen::Matrix4d &pose_inverse,const open3d::camera::PinholeCameraIntrinsic& intrinsic, int dilation_size)
{
    auto depth = std::make_shared<cv::Mat>(cv::Mat::zeros(intrinsic.height_, intrinsic.width_, CV_8UC1));
    if(!cloud.HasPoints()) return depth;
    open3d::geometry::PointCloud points_camera(cloud.points_);
    points_camera.Transform(pose_inverse);
    // std::cout << "points_camera.points_.size(): " << points_camera.points_.size() << std::endl;

    int count = 0;
    for (const Eigen::Vector3d &point: points_camera.points_){
        Eigen::Vector3d point_normalized = point / point[2];
        Eigen::Vector3d uv_homograph = intrinsic.intrinsic_matrix_ * point_normalized;
        int u_ = round(uv_homograph[0]);
        int v_ = round(uv_homograph[1]);
        if(u_ >= 0 && u_ < intrinsic.width_ && v_ >= 0 && v_ < intrinsic.height_){
            depth->at<uint8_t>(v_,u_) = round(point[2] * 1000.0);
            count ++;
        }
    }
    // std::cout << "projected points: " << count << std::endl;

    // Expand depth by dilation
    auto depth_out = std::make_shared<cv::Mat>(cv::Mat::zeros(intrinsic.height_, intrinsic.width_, CV_8UC1));
    // int dilation_size = 5;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    cv::dilate(*depth, *depth_out, element);

    return depth_out;
}

bool create_masked_rgbd(
    const open3d::geometry::Image &rgb, const open3d::geometry::Image &float_depth, const cv::Mat &mask,
    const int &min_points,
    std::shared_ptr<open3d::geometry::RGBDImage> &masked_rgbd)
{
    // auto masked_rgbd = std::make_shared<open3d::geometry::RGBDImage>();
    open3d::geometry::Image masked_depth;
    assert (float_depth.width_ == mask.cols && float_depth.height_ == mask.rows), "depth and mask have different size";
    assert (float_depth.num_of_channels_==1), "depth has more than one channel";
    assert (float_depth.bytes_per_channel_==4), "depth is not in float";
    masked_rgbd->color_ = rgb;

    float CLIP_RATIO = 0.1;
    masked_depth.Prepare(float_depth.width_, float_depth.height_, 1, 4);
    std::vector<float> valid_depth_array;

    for(int v=0; v<float_depth.height_;v++){
        for(int u=0; u<float_depth.width_;u++){
            if(mask.at<uint8_t>(v,u)>0){
                *masked_depth.PointerAt<float>(u,v) = *float_depth.PointerAt<float>(u,v);
                if(*masked_depth.PointerAt<float>(u,v)>0.2) {
                    valid_depth_array.push_back(*masked_depth.PointerAt<float>(u,v));
                }
            }
        }
    }
    if(valid_depth_array.size()<min_points) return false;
    else{
        // sort depth array
        std::sort(valid_depth_array.begin(), valid_depth_array.end());
        // double min_depth_clip = valid_depth_array[std::floor(valid_depth_array.size()*CLIP_RATIO)];
        double max_depth_clip = valid_depth_array[std::ceil(valid_depth_array.size()*(1-CLIP_RATIO))];
        // std::cout<<"["<<min_depth_clip<<","<<max_depth_clip<<"]"<<std::endl;
        masked_depth.ClipIntensity(0.0,max_depth_clip);

        masked_rgbd->depth_ = masked_depth;
        return true;
        // return valid_depth_array.size();
    }
}

O3d_Image_Ptr extract_masked_o3d_image(const O3d_Image &depth, const O3d_Image &mask)
{
    auto masked_depth = std::make_shared<open3d::geometry::Image>();
    masked_depth->Prepare(depth.width_, depth.height_, 1, 4);
    for(int v=0; v<depth.height_;v++){
        for(int u=0; u<depth.width_;u++){
            if(mask.PointerAt<unsigned char>(u,v)>0){
                *masked_depth->PointerAt<float>(u,v) = *depth.PointerAt<float>(u,v);
            }
        }
    }
    return masked_depth;
}


}

}
