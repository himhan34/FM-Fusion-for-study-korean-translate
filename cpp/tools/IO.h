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
                                const std::string &output_file_dir);
    
    bool load_corrs_match_indices(const std::string &corrs_match_indices_file,
                                std::vector<int> &corrs_match_indices);

    bool load_match_results(const std::string &match_file_dir,
                        Eigen::Matrix4d &pose,
                        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                        std::vector<Eigen::Vector3d> &src_centroids,
                        std::vector<Eigen::Vector3d> &ref_centroids,
                        bool verbose=false);
    
    bool save_pose(const std::string &output_dir, const Eigen::Matrix4d &pose);

}

}
#endif
