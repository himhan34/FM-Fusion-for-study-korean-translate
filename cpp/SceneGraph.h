#include <unordered_map>
#include <fstream>

#include "Common.h"
#include "Detection.h"
#include "Instance.h"
#include "Utility.h"

namespace fmfusion
{

class SceneGraph
{
public:
    SceneGraph(const Config &config);

    ~SceneGraph() {};

public:
    void integrate(const int &frame_id,
        const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose,
        std::vector<DetectionPtr> &detections);

    /// @brief  
    /// @param path output sequence folder 
    /// @return 
    bool Save(const std::string &path);

    bool load(const std::string &path);

    const Config &get_config() { return config_; }

protected:
    /// \brief  match vector in [K,1], K is the number of detections;
    /// If detection k is associated, match[k] = matched_instance_id
    Eigen::VectorXi data_association(const std::vector<DetectionPtr> &detections, const std::vector<InstanceId> &active_instances);

    bool create_new_instance(const DetectionPtr &detection,const unsigned int &frame_id,
        const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose);

    std::vector<InstanceId> search_active_instances(const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose);

    void update_active_instances(const std::vector<InstanceId> &active_instances);

    std::unordered_set<InstanceId> recent_instances;

private:
    Config config_;
    InstanceConfig instance_config;
    std::unordered_map<InstanceId,InstancePtr> instance_map;

};


} // namespace fmfusion


