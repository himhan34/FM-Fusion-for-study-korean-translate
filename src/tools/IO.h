#ifndef IO_H_
#define IO_H_
#include <vector>
#include <Common.h>

#include <tools/Utility.h>
#include <sgloop/Graph.h>

namespace fmfusion
{
namespace IO
{
    typedef std::pair<std::string, std::string> RGBDFrameDirs;

    // Read
    bool read_rs_intrinsic(const std::string intrinsic_dir, 
                        open3d::camera::PinholeCameraIntrinsic &intrinsic_);

    bool read_scannet_intrinsic(const std::string intrinsic_folder,
                        open3d::camera::PinholeCameraIntrinsic &intrinsic_);
    bool read_transformation(const std::string &transformation_file, 
                        Eigen::Matrix4d &transformation);

    bool frames_srt_func(const std::string &a, const std::string &b);

    void construct_sorted_frame_table(const std::string &scene_dir,
                                    std::vector<RGBDFrameDirs> &frame_table, 
                                    std::vector<Eigen::Matrix4d> &pose_table);

    bool construct_preset_frame_table(const std::string &root_dir,
                                    const std::string &association_name,
                                    const std::string &trajectory_name,
                                    std::vector<RGBDFrameDirs> &rgbd_table,
                                    std::vector<Eigen::Matrix4d> &pose_table);

    // Write
    void extract_match_instances(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
        const std::vector<fmfusion::NodePtr> &src_nodes,const std::vector<fmfusion::NodePtr> &ref_nodes,
        std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> &match_instances);

    void extract_instance_correspondences(const std::vector<fmfusion::NodePtr> &src_nodes, 
                                        const std::vector<fmfusion::NodePtr> &ref_nodes, 
                                        const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                                        const std::vector<float> &match_scores,
                                        std::vector<Eigen::Vector3d> &src_centroids, 
                                        std::vector<Eigen::Vector3d> &ref_centroids);

    bool save_match_results(const Eigen::Matrix4d &pose,
                        const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                        const std::vector<float> &match_scores,
                        const std::string &output_file_dir);

    bool save_match_results(const float &timestamp,
                        const Eigen::Matrix4d &pose,
                        const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                        const std::vector<Eigen::Vector3d> &src_centroids,
                        const std::vector<Eigen::Vector3d> &ref_centroids,
                        const std::string &output_file_dir);

    bool save_corrs_match_indices(const std::vector<int> &corrs_match_indices,
                                const std::vector<float> &corrs_match_scores,
                                const std::string &output_file_dir);
    
    bool load_corrs_match_indices(const std::string &corrs_match_indices_file,
                                std::vector<int> &corrs_match_indices,
                                std::vector<float> &corrs_match_scores);

    bool load_match_results(const std::string &match_file_dir,
                        float &timestamp,
                        Eigen::Matrix4d &pose,
                        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                        std::vector<Eigen::Vector3d> &src_centroids,
                        std::vector<Eigen::Vector3d> &ref_centroids,
                        bool verbose=false);

    bool load_node_matches(const std::string &match_file_dir,
                        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                        std::vector<bool> &match_tp_masks,
                        std::vector<Eigen::Vector3d> &src_centroids,
                        std::vector<Eigen::Vector3d> &ref_centroids,
                        bool verbose=false);

    bool load_single_col_mask(const std::string &mask_file_dir, std::vector<bool> &mask);

    bool load_pose_file(const std::string &pose_file_dir, Eigen::Matrix4d &pose, bool verbose=false);

    bool read_loop_transformations(const std::string &loop_file_dir,
                        std::vector<LoopPair> &loop_pairs,
                        std::vector<Eigen::Matrix4d> &loop_transformations);

    bool read_loop_pairs(const std::string &loop_file_dir,
                        std::vector<LoopPair> &loop_pairs,
                        std::vector<bool> &loop_tp_masks);

    bool read_frames_poses(const std::string &frame_pose_file,
                    std::unordered_map<std::string, Eigen::Matrix4d> &frame_poses);

    bool read_entire_camera_poses(const std::string &scene_folder,
                        std::unordered_map<std::string, Eigen::Matrix4d> &src_poses_map);

    bool save_pose(const std::string &output_dir, const Eigen::Matrix4d &pose);

    /// \brief  Save the all of the centroids from two graph to the output directory.
    bool save_graph_centroids(const std::vector<Eigen::Vector3d> &src_centroids, 
                            const std::vector<Eigen::Vector3d> &ref_centroids,
                            const std::string &output_dir);

    bool save_instance_info(const std::vector<Eigen::Vector3d> &centroids,
                        const std::vector<std::string> &labels,
                        const std::string &output_dir);
    
    bool load_instance_info(const std::string &instance_info_file,
                        std::vector<Eigen::Vector3d> &centroids,
                        std::vector<std::string> &labels);

    bool write_time(const std::vector<std::string> & header,
                const std::vector<double> & time,
                const std::string & file_name);

}

}
#endif
