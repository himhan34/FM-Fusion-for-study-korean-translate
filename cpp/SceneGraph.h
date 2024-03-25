#include <unordered_map>
#include <fstream>

#include "Common.h"
#include "Color.h"
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

    void merge_overlap_instances(std::vector<InstanceId> instance_list=std::vector<InstanceId>());

    void merge_overlap_structural_instances();

    void extract_point_cloud(const std::vector<InstanceId> instance_list=std::vector<InstanceId>());

    /// \brief  Extract and update bounding box for each instance.
    void extract_bounding_boxes();

    void remove_invalid_instances();

    /// \brief  Get geometries for each instance.
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> get_geometries(bool point_cloud=true, bool bbox=false);

    bool is_empty() { return instance_map.empty(); }

    void Transform(const Eigen::Matrix4d &pose);

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

    int create_new_instance(const DetectionPtr &detection,const unsigned int &frame_id,
        const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose);

    std::vector<InstanceId> search_active_instances(const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose, const double search_radius=5.0);

    void update_active_instances(const std::vector<InstanceId> &active_instances);

    bool IsSemanticSimilar(const std::unordered_map<std::string,float> &measured_labels_a,
        const std::unordered_map<std::string,float> &measured_labels_b);
    
    /// \brief  Compute the 2D IoU (horizontal plane) between two oriented bounding boxes.
    double Compute2DIoU(const open3d::geometry::OrientedBoundingBox &box_a, const open3d::geometry::OrientedBoundingBox &box_b);

    /// \brief  Compute the 3D IoU between two point clouds.
    /// \param cloud_a, point cloud of the larger instance
    /// \param cloud_b, point cloud of the smaller instance
    double Compute3DIoU(const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation=1.0);

    std::unordered_set<InstanceId> recent_instances;

private:
    Config config_;
    InstanceConfig instance_config;
    std::unordered_map<InstanceId,InstancePtr> instance_map;
    std::unordered_map<std::string, std::vector<InstanceId>> label_instance_map;
    InstanceId latest_created_instance_id;
    int last_cleanup_frame_id;

};


} // namespace fmfusion


