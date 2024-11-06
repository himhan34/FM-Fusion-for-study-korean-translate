#ifndef FMFUSION_INSTANCE_H
#define FMFUSION_INSTANCE_H

#include <list>
#include <string>

#include "open3d/Open3D.h"
#include "Detection.h"
#include "Common.h"
#include "SubVolume.h"

namespace fmfusion {

    namespace o3d_utility = open3d::utility;

    class Instance {

    public:
        Instance(const InstanceId id, const unsigned int frame_id, const InstanceConfig &config);

        ~Instance() {};

        void init_bayesian_fusion(const std::vector<std::string> &label_set);
    public:
        void integrate(const int &frame_id,
                       const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose);

        // Record measured labels and update the predicted label.
        void update_label(const DetectionPtr &detection);

        /// \brief  Update the centroid from volume units origins.
        void fast_update_centroid() { centroid = volume_->get_centroid(); };

        bool update_point_cloud(int cur_frame_id, int min_frame_gap = 10);

        /// @brief  add the probability vector from each measurement.
        ///         And update the predicted label.
        /// @param probability_vector_ computed from BayesianLabel
        /// @return success flag
        bool update_semantic_probability(const Eigen::VectorXf &probability_vector_);

        void merge_with(const O3d_Cloud_Ptr &other_cloud,
                        const std::unordered_map<std::string, float> &label_measurements, 
                        const int &observations_);

        void extract_write_point_cloud();
    
        void filter_pointcloud_statistic();

        bool filter_pointcloud_by_cluster();

        void CreateMinimalBoundingBox();

        /// \brief  label_str:label_name(score),label_name(score),...
        ///         record previouse measured labels
        void load_previous_labels(const std::string &labels_str);

        void load_obser_count(const int &obs_count){
            observation_count = obs_count;
        }

        void save(const std::string &path);

        void load(const std::string &path);

    public:
        SubVolume *get_volume() { return volume_; }

        LabelScore get_predicted_class() const { 
            return predicted_label;
        }

        std::unordered_map<std::string, float> get_measured_labels() const { 
            return measured_labels; 
        }

        size_t get_cloud_size() const;

        O3d_Cloud_Ptr get_complete_cloud() const;

        InstanceConfig get_config() const { return config_; }

        std::string get_measured_labels_string() const {
            std::stringstream label_measurements;
            for (const auto &label_score: measured_labels) {
                label_measurements << label_score.first
                                   << "(" << std::fixed << std::setprecision(2) << label_score.second << "),";
            }
            return label_measurements.str();
        }

        int get_observation_count() const {
            return observation_count;
        }

        InstanceId get_id() const { return id_; }

        void change_id(InstanceId new_id) { 
            id_ = new_id; 
        }

        O3d_Cloud_Ptr get_point_cloud() const;

        bool isBayesianFusion() const { return bayesian_label; }

    private:
        /// @brief Extract the label with the maximum probability
        ///        as the predicted class.
        void extract_bayesian_prediciton();

    public:
        unsigned int frame_id_; // latest integration frame id
        unsigned int update_frame_id; // update point cloud and bounding box
        Eigen::Vector3d color_;
        std::shared_ptr<cv::Mat> observed_image_mask; // Poject volume on image plane;
        // open3d::pipelines::integration::InstanceTSDFVolume *volume_;
        SubVolume *volume_;
        O3d_Cloud_Ptr point_cloud;
        Eigen::Vector3d centroid;
        Eigen::Vector3d normal;
        std::shared_ptr<open3d::geometry::OrientedBoundingBox> min_box;

    private:
        InstanceId id_; // >=1
        InstanceConfig config_;
        std::unordered_map<std::string, float> measured_labels;
        LabelScore predicted_label;
        int observation_count;
        O3d_Cloud_Ptr merged_cloud;

        // For Bayesian likelihood
        std::vector<std::string> semantic_labels;
        Eigen::VectorXf probability_vector;
        bool bayesian_label;
    };

    typedef std::shared_ptr<Instance> InstancePtr;

} // namespace fmfusion

#endif // FMFUSION_INSTANCE_H

    