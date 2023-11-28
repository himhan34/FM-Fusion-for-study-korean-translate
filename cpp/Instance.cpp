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
    min_box = std::make_shared<open3d::geometry::OrientedBoundingBox>();
    predicted_label = std::make_pair("unknown",0.0);

    std::srand(std::time(nullptr));
    color_ = Eigen::Vector3d((double)rand() / RAND_MAX,(double)rand() / RAND_MAX,(double)rand() / RAND_MAX);
    centroid = Eigen::Vector3d(0.0,0.0,0.0);
    observation_count = 1;

}

void Instance::integrate(const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose)
{
    // open3d::utility::LogInfo("Integrate Instance");
    volume_->Integrate(*rgbd_image,config_.intrinsic,pose);
}

void Instance::filter_pointcloud_statistic()
{
    if(point_cloud){
        size_t old_points_number = point_cloud->points_.size();
        O3d_Cloud_Ptr output_cloud;
        std::tie(output_cloud, std::ignore) = point_cloud->RemoveStatisticalOutliers(20,2.0);
        point_cloud = output_cloud;
        // std::cout<<"!!!!Filter point cloud from "<<old_points_number<<" to "<<point_cloud->points_.size()<<std::endl;
    }
}

bool Instance::filter_pointcloud_by_cluster()
{
    if(point_cloud==nullptr){
        return false;
    }
    else{
        auto labels = point_cloud->ClusterDBSCAN(config_.cluster_eps,config_.cluster_min_points,true);
        unordered_map<int,int> cluster_counts; // cluster_id:count
        const size_t old_points_number = point_cloud->points_.size();

        // create cluster
        for(auto label:labels){
            if(cluster_counts.find(label)==cluster_counts.end()){
                cluster_counts[label] = 1;
            }
            else{
                cluster_counts[label] += 1;
            }
        }

        // find invalid cluster
        std::unordered_set<int> invalid_cluster({-1});  

        // filter 
        size_t k =0;
        for(size_t i=0;i<old_points_number;i++){
            if(invalid_cluster.find(labels[i])!=invalid_cluster.end()){ //invalid
                continue;
            }
            else{ //valid 
                point_cloud->points_[k] = point_cloud->points_[i];
                if(point_cloud->HasNormals()) point_cloud->normals_[k] = point_cloud->normals_[i];
                if(point_cloud->HasCovariances()) point_cloud->covariances_[k] = point_cloud->covariances_[i];
                if(point_cloud->HasColors()) point_cloud->colors_[k] = point_cloud->colors_[i];
                k++;
            }
        }
        point_cloud->points_.resize(k);
        point_cloud->PaintUniformColor(color_);
        o3d_utility::LogInfo("Filter point cloud from {} to {}.",old_points_number,k);
        return true;
    }
}

void Instance::CreateMinimalBoundingBox()
{
    if(point_cloud->points_.size()<10){
        return;
    }

    using namespace open3d::geometry;
    std::shared_ptr<TriangleMesh> mesh;
    std::tie(mesh, std::ignore) = point_cloud->ComputeConvexHull(false);
    double min_vol = -1;
    min_box->Clear();
    PointCloud hull_pcd;
    for (auto& tri : mesh->triangles_) {
        hull_pcd.points_ = mesh->vertices_;
        Eigen::Vector3d a = mesh->vertices_[tri(0)];
        Eigen::Vector3d b = mesh->vertices_[tri(1)];
        Eigen::Vector3d c = mesh->vertices_[tri(2)];
        Eigen::Vector3d u = b - a;
        Eigen::Vector3d v = c - a;
        Eigen::Vector3d w = u.cross(v);
        v = w.cross(u);
        u = u / u.norm();
        v = v / v.norm();
        w = w / w.norm();
        Eigen::Matrix3d m_rot;
        // m_rot << u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2];
        // set roll and pitch to zero
        m_rot << u[0], v[0], 0, u[1], v[1], 0, 0, 0, 1;
        hull_pcd.Rotate(m_rot.inverse(), a);

        const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
        double volume = aabox.Volume();
        if (min_vol == -1. || volume < min_vol) {
            min_vol = volume;
            *min_box = aabox.GetOrientedBoundingBox();
            min_box->Rotate(m_rot, a);
        }
    }
    min_box->color_ = color_;

    //
    if(predicted_label.first=="floor"){
        min_box->center_[2] = min_box->center_[2] - min_box->extent_[2]/2 + 0.05;
        min_box->extent_[2] = 0.02;
    }
}

void Instance::merge_with(const O3d_Cloud_Ptr &other_cloud, 
    const std::unordered_map<std::string,float> &label_measurements, const int &observations_)
{
    // integrate point cloud
    *point_cloud += *other_cloud;
    auto ds_cloud = point_cloud->VoxelDownSample(config_.voxel_length);
    point_cloud->Clear();
    *point_cloud = *ds_cloud;
    point_cloud->PaintUniformColor(color_);

    // update labels
    for(const auto label_score:label_measurements){
        if(measured_labels.find(label_score.first)==measured_labels.end()){
            measured_labels[label_score.first] = label_score.second;
        }
        else{
            measured_labels[label_score.first] += label_score.second;
        }
        // update prediction
        if(measured_labels[label_score.first]>predicted_label.second){
            predicted_label = std::make_pair(label_score.first,measured_labels[label_score.first]);
        }

    }

    observation_count += observations_;

    //
    CreateMinimalBoundingBox();

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
    double voxel_weight_threshold = config_.min_voxel_weight * observation_count;
    auto cloud = volume_->ExtractWeightedPointCloud(std::min(std::max(voxel_weight_threshold,2.0),4.0));
    // auto cloud = volume_->ExtractWeightedPointCloud(2.0);


    centroid = cloud->GetCenter();
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
        if(score>predicted_label.second){
            predicted_label = std::make_pair(label,score);
        }
        // cout<<label<<":"<<score<<",";
    }
    // cout<<"\n";
}

}