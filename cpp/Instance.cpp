#include "Instance.h"


namespace fmfusion
{


Instance::Instance(const InstanceId id, const unsigned int frame_id, const InstanceConfig &config): 
    id_(id), frame_id_(frame_id),config_(config)
{
    // open3d::utility::LogInfo("Initialize Instance");
    volume_ = new open3d::pipelines::integration::InstanceTSDFVolume(
        config_.voxel_length,config_.sdf_trunc,open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
    point_cloud = nullptr;
    predicted_label = std::make_pair("unknown",0.0);

    color_ = Eigen::Vector3d((double)rand() / RAND_MAX,(double)rand() / RAND_MAX,(double)rand() / RAND_MAX);
    centroid = Eigen::Vector3d(0.0,0.0,0.0);
    observation_count = 1;

}

void Instance::integrate(const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose)
{
    // open3d::utility::LogInfo("Integrate Instance");
    volume_->Integrate(*rgbd_image,config_.intrinsic,pose);
}

void Instance::update_label(const DetectionPtr &detection)
{
    for (const auto &label_score: detection->labels_){
        if(measured_labels.find(label_score.first) == measured_labels.end()){
            measured_labels[label_score.first] = label_score.second;
        }
        else{
            measured_labels[label_score.first] += label_score.second;
        }

        // Update prediction
        if (measured_labels[label_score.first]>predicted_label.second){
            predicted_label = std::make_pair(label_score.first,measured_labels[label_score.first]);
        }

    }
    observation_count ++;
}

std::shared_ptr<open3d::geometry::PointCloud> Instance::extract_point_cloud()
{
    auto cloud = volume_->ExtractPointCloud();


    cloud->PaintUniformColor(color_);

    return cloud;
}

void Instance::load_previous_labels(const std::string &labels_str)
{
    std::stringstream ss(labels_str);
    std::string label_score;

    while (std::getline(ss, label_score, ',')) {
        std::stringstream ss2(label_score); // label_name(score)
        std::string label;
        float score;
        std::getline(ss2, label, '(');
        ss2 >> score;
        measured_labels[label] = score;
        // cout<<label<<":"<<score<<",";
    }
    // cout<<"\n";
}

}