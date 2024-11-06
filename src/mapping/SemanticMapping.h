#ifndef FMFUSION_SEMANTICMAPPING_H
#define FMFUSION_SEMANTICMAPPING_H

#include <unordered_map>
#include <fstream>

#include "Common.h"
#include "tools/Color.h"
#include "tools/Utility.h"
#include "Instance.h"
#include "SemanticDict.h"
#include "BayesianLabel.h"

namespace fmfusion {

    class SemanticMapping {
    public:
        SemanticMapping(const MappingConfig &mapping_cfg, const InstanceConfig &instance_cfg);

        ~SemanticMapping() {};

    public:
        void integrate(const int &frame_id,
                       const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose,
                       std::vector<DetectionPtr> &detections);

        int merge_overlap_instances(std::vector<InstanceId> instance_list = std::vector<InstanceId>());

        int merge_floor(bool verbose=false);

        int merge_overlap_structural_instances(bool merge_all = true);

        int merge_other_instances(std::vector<InstancePtr> &instances);
        
        void extract_point_cloud(const std::vector<InstanceId> instance_list = std::vector<InstanceId>());

        /// \brief  Extract and update bounding box for each instance.
        void extract_bounding_boxes();

        int update_instances(const int &cur_frame_id, const std::vector<InstanceId> &instance_list);

        /// @brief  clean all and update all.
        void refresh_all_semantic_dict();

        std::shared_ptr<open3d::geometry::PointCloud> export_global_pcd(bool filter = false, float vx_size = -1.0);

        std::vector<Eigen::Vector3d> export_instance_centroids(int earliest_frame_id = -1) const;

        std::vector<std::string> export_instance_annotations(int earliest_frame_id = -1) const;

        bool query_instance_info(const std::vector<InstanceId> &names,
                                 std::vector<Eigen::Vector3f> &centroids,
                                 std::vector<std::string> &labels);

        void remove_invalid_instances();

        /// \brief  Get geometries for each instance.
        std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
        get_geometries(bool point_cloud = true, bool bbox = false);

        bool is_empty() { return instance_map.empty(); }

        InstancePtr get_instance(const InstanceId &name) { return instance_map[name]; }

        void Transform(const Eigen::Matrix4d &pose);

        /// @brief
        /// @param path output sequence folder
        /// @return
        bool Save(const std::string &path);

        bool load(const std::string &path);

        /// \brief  Export instances to the vector.
        ///         Filter instances that are too small or not been observed for a long time.
        void export_instances(std::vector<InstanceId> &names, std::vector<InstancePtr> &instances,
                              int earliest_frame_id = 0);

    protected:
        /// \brief  match vector in [K,1], K is the number of detections;
        /// If detection k is associated, match[k] = matched_instance_id
        int
        data_association(const std::vector<DetectionPtr> &detections, const std::vector<InstanceId> &active_instances,
                         Eigen::VectorXi &matches,
                         std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs);

        int create_new_instance(const DetectionPtr &detection, const unsigned int &frame_id,
                                const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image,
                                const Eigen::Matrix4d &pose);

        std::vector<InstanceId> search_active_instances(const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose,
                                                        const double search_radius = 5.0);

        void update_active_instances(const std::vector<InstanceId> &active_instances);

        void update_recent_instances(const int &frame_id,
                                     const std::vector<InstanceId> &active_instances,
                                     const std::vector<InstanceId> &new_instances);

        bool IsSemanticSimilar(const std::unordered_map<std::string, float> &measured_labels_a,
                               const std::unordered_map<std::string, float> &measured_labels_b);

        /// \brief  Compute the 2D IoU (horizontal plane) between two oriented bounding boxes.
        double Compute2DIoU(const open3d::geometry::OrientedBoundingBox &box_a,
                            const open3d::geometry::OrientedBoundingBox &box_b);

        /// \brief  Compute the 3D IoU between two point clouds.
        /// \param cloud_a, point cloud of the larger instance
        /// \param cloud_b, point cloud of the smaller instance
        double Compute3DIoU(const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation = 1.0);

        int merge_ambiguous_instances(const std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs);

        // Recent observed instances
        std::unordered_set<InstanceId> recent_instances;

    private:
        // Config config_;
        MappingConfig mapping_config;
        InstanceConfig instance_config;
        std::unordered_map<InstanceId, InstancePtr> instance_map;
        std::unordered_map<std::string, std::vector<InstanceId>> label_instance_map;
        SemanticDictServer semantic_dict_server;
        BayesianLabel *bayesian_label;

        InstanceId latest_created_instance_id;
        int last_cleanup_frame_id;
        int last_update_frame_id;

    };


} // namespace fmfusion


#endif //FMFUSION_SEMANTICMAPPING_H
