#include "SceneGraph.h"

namespace fmfusion
{

SceneGraph::SceneGraph(const Config &config): config_(config)
{
    open3d::utility::LogInfo("Initialize SceneGraph server");
    instance_config.voxel_length = config_.voxel_length;
    instance_config.sdf_trunc = config_.sdf_trunc;
    instance_config.intrinsic = config_.intrinsic;

}

void SceneGraph::integrate(const int &frame_id,
    const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose,
    std::vector<DetectionPtr> &detections)
{
    open3d::utility::LogInfo("## Integrate SceneGraph");
    int n_det = detections.size();

    if(n_det>1){
        o3d_utility::Timer timer_query, timer_da, timer_integrate;
        timer_query.Start();
        auto depth_cloud = O3d_Cloud::CreateFromDepthImage(rgbd_image->depth_,config_.intrinsic,pose.inverse(),1.0,config_.depth_max);
        auto active_instances = search_active_instances(depth_cloud,pose);
        timer_query.Stop();

        timer_da.Start();
        auto matches = data_association(detections, active_instances);
        timer_da.Stop();
        // std::cout<<"matches: "<<matches.transpose()<<std::endl;

        timer_integrate.Start();
        for (int k_=0;k_<n_det;k_++){
            auto masked_rgbd = std::make_shared<open3d::geometry::RGBDImage>();
            int valid_points_number = utility::create_masked_rgbd(
                rgbd_image->color_,rgbd_image->depth_,detections[k_]->instances_idxs_,masked_rgbd);

            if(valid_points_number<config_.min_instance_masks) {
                matches(k_) = -1;
                continue;
            }

            if (matches(k_)>0){
                auto matched_instance = instance_map[matches(k_)];
                matched_instance->integrate(masked_rgbd,pose.inverse());
                matched_instance->update_label(detections[k_]);
            }
            else{
                create_new_instance(detections[k_], frame_id,masked_rgbd,pose);
            }
        }
        timer_integrate.Stop();

        // Visualize associations
        auto rgb_cv = std::make_shared<cv::Mat>(rgbd_image->color_.height_,rgbd_image->color_.width_,CV_8UC3);
        memcpy(rgb_cv->data,rgbd_image->color_.data_.data(),rgbd_image->color_.data_.size()*sizeof(uint8_t));
        cv::cvtColor(*rgb_cv,*rgb_cv,cv::COLOR_RGB2BGR);

        std::stringstream ss, msg;
        ss<<config_.tmp_dir<<"/tmp/frame-"<<std::fixed<<std::setw(6)<<std::setfill('0')<<frame_id<<".png";
        std::unordered_map<InstanceId,CvMatPtr> active_instance_masks;
        std::unordered_map<InstanceId,Eigen::Vector3d> active_instance_colors;
        msg<<"active instances: ";
        for(auto &instance_j:instance_map){
            if(instance_j.second->observed_image_mask){
                active_instance_masks.emplace(instance_j.first,instance_j.second->observed_image_mask);
                active_instance_colors.emplace(instance_j.first,instance_j.second->color_);
                msg<<instance_j.second->get_predicted_class().first<<", ";
            }
        }
        // std::cout<<msg.str()<<"\n";

        if(config_.save_da_images){
            auto render_img = utility::RenderDetections(rgb_cv,detections,active_instance_masks,matches, active_instance_colors);
            cv::imwrite(ss.str(),*render_img);
        }
        //
        update_active_instances(active_instances);
        
        // Manage recent observed instances
        for(InstanceId j_:active_instances){
            recent_instances.emplace(j_);
        }

        o3d_utility::LogInfo("## {}/{} instances",active_instances.size(),instance_map.size());
        o3d_utility::LogInfo("Time record (ms): query {:f}, da {:f}, integrate {:f}",
            timer_query.GetDurationInMillisecond(),timer_da.GetDurationInMillisecond(),timer_integrate.GetDurationInMillisecond());
    }

}

std::vector<InstanceId> SceneGraph::search_active_instances(
    const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose)
{
    std::vector<InstanceId> active_instances;
    const size_t MIN_UNITS= 1;

    for(auto &instance_j:instance_map){
        Eigen::Vector3d centroid_cam_j = pose.block<3,3>(0,0)*instance_j.second->centroid+pose.block<3,1>(0,3);
        if(centroid_cam_j(2)<0.1) continue;

        auto observed_cloud = std::make_shared<O3d_Cloud>();
        instance_j.second->get_volume()->query_observed_points(depth_cloud,observed_cloud);
        if(observed_cloud->points_.size()>config_.min_instance_masks){
            std::shared_ptr<cv::Mat> instance_img_mask = utility::PrjectionCloudToDepth(
                *observed_cloud,pose.inverse(),instance_config.intrinsic,config_.dilation_size);
            instance_j.second->observed_image_mask = instance_img_mask;
            active_instances.emplace_back(instance_j.first);
        }
    }

    return active_instances;
}

void SceneGraph::update_active_instances(const std::vector<InstanceId> &active_instances)
{
    for(InstanceId j_:active_instances){
        auto instance_j = instance_map[j_];
        instance_j->observed_image_mask.reset();
        instance_j->fast_update_centroid();
    }
}

Eigen::VectorXi SceneGraph::data_association(const std::vector<DetectionPtr> &detections,
    const std::vector<InstanceId> &active_instances)
{
    int K = detections.size();
    int M = active_instances.size();

    Eigen::VectorXi matches = Eigen::VectorXi::Zero(K);
    if (M<1) return matches;

    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(K,M);
    Eigen::MatrixXi assignment = Eigen::MatrixXi::Zero(K,M);
    Eigen::MatrixXi assignment_colwise = Eigen::MatrixXi::Zero(K,M);
    Eigen::MatrixXi assignment_rowise = Eigen::MatrixXi::Zero(K,M);

    for (int k_=0;k_<K;k_++){
        const auto &zk = detections[k_];
        double zk_area = double(cv::countNonZero(zk->instances_idxs_));
        for (int m_=0;m_<M;m_++){
            auto instance_m = instance_map[active_instances[m_]];
            cv::Mat overlap = instance_m->observed_image_mask->mul(zk->instances_idxs_);
            double overlap_area = double(cv::countNonZero(overlap));
            double instance_area = double(cv::countNonZero(*instance_m->observed_image_mask));
            iou(k_,m_) = overlap_area/(zk_area+instance_area-overlap_area);
            // iou(k_,m_) = double(cv::countNonZero(overlap))/double(cv::countNonZero(*instance_m->observed_image_mask));   // overlap/r_m
        }
    } 

    // Find the maximum match for each colume
    for (int m_=0;m_<M;m_++){
        int max_row;
        double max_iou = iou.col(m_).maxCoeff(&max_row);
        if (max_iou>config_.min_iou)assignment_colwise(max_row,m_) = 1;
    }

    // Find the maximum match for each row
    for (int k_=0;k_<K;k_++){
        int max_col;
        double max_iou = iou.row(k_).maxCoeff(&max_col);
        if (max_iou>config_.min_iou)assignment_rowise(k_,max_col) = 1;
    }

    assignment = assignment_colwise + assignment_rowise;

    // export matches
    int count= 0;
    for (int k_=0;k_<K;k_++){
        for (int m_=0;m_<M;m_++){
            if (assignment(k_,m_)==2){
                matches(k_) = active_instances[m_];
                count ++;
                break;
            }
        }
    }

    // std::cout<<iou<<std::endl;   
    // std::cout<<assignment<<std::endl;
    o3d_utility::LogInfo("{}/({},{}) associations out of detections and active instances.",count,K,M);

    return matches;
}

bool SceneGraph::create_new_instance(const DetectionPtr &detection, const unsigned int &frame_id,
    const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose)
{
    // open3d::utility::LogInfo("Create new instance");
    auto instance = std::make_shared<Instance>(instance_map.size()+1,frame_id,instance_config);
    instance->integrate(rgbd_image,pose.inverse());
    instance->update_label(detection);
    instance->fast_update_centroid();
    instance_map.emplace(instance->id_,instance);

    return true;
}

bool SceneGraph::Save(const std::string &path)
{
    using namespace o3d_utility::filesystem;
    if(!DirectoryExists(path)) MakeDirectory(path);

    open3d::geometry::PointCloud global_instances_pcd;

    typedef std::pair<InstanceId,std::string> InstanceInfo;
    std::vector<InstanceInfo> instance_info;

    for (const auto &instance: instance_map){
        LabelScore semantic_class_score = instance.second->get_predicted_class();
        auto instance_cloud = instance.second->extract_point_cloud();
        if(instance_cloud->points_.size()<10) continue;

        global_instances_pcd += *instance_cloud;
        stringstream ss; // instance info string
        ss<<std::setw(4)<<std::setfill('0')<<instance.second->id_;
        open3d::io::WritePointCloud(path+"/"+ss.str()+".ply",*instance_cloud);

        ss<<";"
            <<semantic_class_score.first<<"("<<std::fixed<<std::setprecision(2)<<semantic_class_score.second<<")"<<";"
            <<instance.second->get_observation_count()<<";"
            <<instance.second->get_label_measurements()<<";"
            <<instance_cloud->points_.size()<<";\n";

        instance_info.emplace_back(instance.second->id_,ss.str());
        o3d_utility::LogInfo("Instance {:s} has {:d} points",semantic_class_score.first, instance_cloud->points_.size());
    }

    // Sort instance info and write it to text 
    std::ofstream ofs(path+"/instance_info.txt",std::ofstream::out);
    ofs<<"# instance_id;semantic_class(aggregate_score);observation_count;label_measurements;points_number\n";
    std::sort(instance_info.begin(),instance_info.end(),[](const InstanceInfo &a, const InstanceInfo &b){
        return a.first<b.first;
    });
    for (const auto &info:instance_info){
        ofs<<info.second;
    }
    ofs.close();

    // Save global instance map
    if(global_instances_pcd.points_.size()<1) return false;

    open3d::io::WritePointCloud(path+"/instance_map.ply",global_instances_pcd);
    o3d_utility::LogWarning("Save semantic instances to {:s}",path);

    return true;
}

bool SceneGraph::load(const std::string &path)
{

    o3d_utility::LogInfo("Load SceneGraph from {:s}",path);
    using namespace o3d_utility::filesystem;
    if(!DirectoryExists(path)) return false;

    // Load instance info
    std::ifstream ifs(path+"/instance_info.txt",std::ifstream::in);
    std::string line;
    std::getline(ifs,line); // skip header
    while(std::getline(ifs,line)){
        std::stringstream ss(line);
        std::string instance_id_str;
        std::getline(ss,instance_id_str,';');
        InstanceId instance_id = std::stoi(instance_id_str);
        std::string label_score_str, observ_str, label_measurments_str, observation_count_str;
        std::getline(ss,label_score_str,';');
        std::getline(ss,observ_str,';');
        std::getline(ss,label_measurments_str,';');
        // std::getline(ss,observation_count_str,')');
        
        InstancePtr instance_toadd = std::make_shared<Instance>(instance_id,10,instance_config);
        instance_toadd->load_previous_labels(label_measurments_str);
        instance_toadd->point_cloud = open3d::io::CreatePointCloudFromFile(path+"/"+instance_id_str+".ply");
        instance_map.emplace(instance_id,instance_toadd);

        // cout<<instance_id_str<<","<<label_measurments_str
        //     <<","<<cloud->points_.size()
        //     <<"\n";
    
    }

    o3d_utility::LogInfo("Load {:d} instances",instance_map.size());

    return true;

}

}