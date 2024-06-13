//
// Created by qzj on 6/8/24.
//

#ifndef OPEN3DEXTRA_G3REG_API_H
#define OPEN3DEXTRA_G3REG_API_H

#include "back_end/reglib.h"
#include "mapping/Instance.h"
#include "robot_utils/eigen_types.h"

using VoxelKey = std::tuple<int, int, int>;

class Voxel3D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Voxel3D> Ptr;

    explicit Voxel3D(const VoxelKey key) : key_(key) {}

    // Returns the number of score-index pairs
    int count() const {
        return score_index_pairs_.size();
    }

    // Sorts the score-index pairs in descending order of scores
    void sort() {
        std::sort(score_index_pairs_.begin(), score_index_pairs_.end(),
                  [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                      return a.first > b.first; // Descending order
                  });
    }

    // Returns the top K score-index pairs
    std::vector<std::pair<float, int>> getTopK(int k) const {
        if (k >= score_index_pairs_.size()) {
            return score_index_pairs_;
        }
        return std::vector<std::pair<float, int>>(score_index_pairs_.begin(), score_index_pairs_.begin() + k);
    }

    // Inserts a new score-index pair
    void insertPoint(int i, float score) {
        score_index_pairs_.emplace_back(score, i);
    }

private:
    VoxelKey key_;
    std::vector<std::pair<float, int>> score_index_pairs_;
};


using VoxelMap3D = std::unordered_map<VoxelKey, Voxel3D::Ptr, robot_utils::Vec3dHash>;

template<typename T>
VoxelKey point_to_voxel_key(const Eigen::Matrix<T, 3, 1> &point, const float &voxel_size) {
    int x = static_cast<int>(std::floor(point.x() / voxel_size));
    int y = static_cast<int>(std::floor(point.y() / voxel_size));
    int z = static_cast<int>(std::floor(point.z() / voxel_size));
    return std::make_tuple(x, y, z);
}

class G3RegAPI {
public:
    g3reg::Config cfg_;
    VoxelMap3D voxel_map;

    G3RegAPI(g3reg::Config &cfg) : cfg_(cfg) {

    }

    void estimate_pose(const std::vector<fmfusion::NodePtr> &src_nodes,
                       const std::vector<fmfusion::NodePtr> &ref_nodes,
                       const std::vector<std::pair<uint32_t, uint32_t>> &match_pairs,
                       const std::vector<float> &corr_scores_vec,
                       const std::vector<Eigen::Vector3d> &corr_src_points,
                       const std::vector<Eigen::Vector3d> &corr_ref_points,
                       fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                       fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr,
                       Eigen::Matrix4d &pose) {
        std::stringstream msg;
        // assert(src_corrp.rows() == ref_corrp.rows());

//    稠密点匹配
        std::vector<int> indices = downsample_corr_indices(corr_src_points, corr_scores_vec);
//    合并instance中心匹配和稠密点匹配
        Eigen::MatrixX3d src_corrp(match_pairs.size() + indices.size(), 3);
        Eigen::MatrixX3d ref_corrp(match_pairs.size() + indices.size(), 3);
        for (int i = 0; i < indices.size(); i++) {
            src_corrp.row(i) = corr_src_points[indices[i]];
            src_corrp.row(i) = corr_ref_points[indices[i]];
        }
        msg << "Correspondence points: " << corr_src_points.size() << " -> " << indices.size() << "\n";

//    中心匹配
        msg << match_pairs.size() << " Matched instance pairs: \n";
        for (int i = 0; i < match_pairs.size(); i++) {
            const auto &pair = match_pairs[i];
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            src_corrp.row(i + indices.size()) = src_node->centroid;
            ref_corrp.row(i + indices.size()) = ref_node->centroid;

            // Add raw point clouds，用于验证
            MergePointClouds(src_cloud_ptr, src_node->cloud);
            MergePointClouds(ref_cloud_ptr, ref_node->cloud);

            msg << "(" << pair.first << "," << pair.second << ") "
                << "(" << src_node->semantic << "," << ref_node->semantic << ")\n";
        }
        std::cout << msg.str() << std::endl;

//    求解位姿矩阵
        auto src_cloud = vectorToMatrix(src_cloud_ptr->points_);
        auto ref_cloud = vectorToMatrix(ref_cloud_ptr->points_);
        FRGresult result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud, ref_cloud, cfg_);
        std::cout << "FRG result: " << std::endl;
        std::cout << "verify time: " << result.verify_time << std::endl;
        std::cout << "total time: " << result.total_time << std::endl;
        std::cout << "tf: \n" << result.tf << std::endl;
        pose = result.tf;
    };

    std::vector<int> downsample_corr_indices(const std::vector<Eigen::Vector3d> &corr_src_points,
                                             const std::vector<float> &corr_scores_vec, int target = 1000) {
        if (target >= corr_src_points.size()) {
            std::vector<int> indices;
            for (int i = 0; i < corr_src_points.size(); i++) {
                indices.push_back(i);
            }
            return indices;
        }

        cutCloud(corr_src_points, corr_scores_vec, 1.0, voxel_map);

        int max_num = 0;
        for (const auto &voxel_pair: voxel_map) {
            auto &voxel = voxel_pair.second;
            voxel->sort();
            max_num = std::max(max_num, voxel->count());
        }

        int left = 1, right = 1;
        std::vector<int> indices;
        indices.reserve(corr_src_points.size());
        while (left <= right) {
            int mid = left + (right - left) / 2;
            indices.clear();
            for (const auto &voxel_pair: voxel_map) {
                auto &voxel = voxel_pair.second;
                const auto &top_k = voxel->getTopK(mid);
                for (size_t i = 0; i < top_k.size(); ++i) {
                    indices.push_back(top_k[i].second);
                }
            }

            if (indices.size() < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
//        for (int i = 0; i < indices.size(); i++) {
//            std::cout << indices[i] << " ";
//        }
//        std::cout << std::endl;
        return indices;
    }

    void cutCloud(
            const std::vector<Eigen::Vector3d> &corr_src_points,
            const std::vector<float> &corr_scores_vec,
            const float &voxel_size, VoxelMap3D &voxel_map) {
        for (size_t i = 0; i < corr_src_points.size(); i++) {
            VoxelKey position = point_to_voxel_key(corr_src_points[i], voxel_size);
            VoxelMap3D::iterator voxel_iter = voxel_map.find(position);
            if (voxel_iter != voxel_map.end()) {
                voxel_iter->second->insertPoint(i, corr_scores_vec[i]);
            } else {
                Voxel3D::Ptr voxel = Voxel3D::Ptr(new Voxel3D(position));
                voxel->insertPoint(i, corr_scores_vec[i]);
                voxel_map.insert(std::make_pair(position, voxel));
            }
        }
    }

    Eigen::MatrixX3d vectorToMatrix(const std::vector<Eigen::Vector3d> &vector) {
        // 创建一个动态大小的矩阵，行数为 vector 的大小，每行3列
        Eigen::MatrixX3d matrix(vector.size(), 3);

        // 将 vector 中的每个点复制到矩阵的对应行中
        for (size_t i = 0; i < vector.size(); ++i) {
            matrix.row(i) = vector[i];
        }

        return matrix;
    }

    // Function to merge cloud2_ptr into cloud1_ptr
    void MergePointClouds(fmfusion::O3d_Cloud_Ptr &cloud1_ptr, const fmfusion::O3d_Cloud_Ptr &cloud2_ptr) {

        // Concatenate points
        cloud1_ptr->points_.insert(cloud1_ptr->points_.end(),
                                   cloud2_ptr->points_.begin(), cloud2_ptr->points_.end());

        // Concatenate colors if both clouds have colors
        if (cloud2_ptr->HasColors()) {
            cloud1_ptr->colors_.insert(cloud1_ptr->colors_.end(),
                                       cloud2_ptr->colors_.begin(), cloud2_ptr->colors_.end());
        }

        // Concatenate normals if both clouds have normals
        if (cloud2_ptr->HasNormals()) {
            cloud1_ptr->normals_.insert(cloud1_ptr->normals_.end(),
                                        cloud2_ptr->normals_.begin(), cloud2_ptr->normals_.end());
        }
    }


    Eigen::Matrix4d
    icp_refine(fmfusion::O3d_Cloud_Ptr &src_cloud_ptr, fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr,
               Eigen::Matrix4d pred_pose = Eigen::Matrix4d::Identity(),
               float max_correspondence_distance = 0.5, float voxel_size = 0.1) {
        using namespace open3d;

//        ref_cloud_ptr->EstimateNormals(geometry::KDTreeSearchParamHybrid(0.5, 30));
        auto downsampled_cloud1_ptr = src_cloud_ptr->VoxelDownSample(voxel_size);
        auto downsampled_cloud2_ptr = ref_cloud_ptr->VoxelDownSample(voxel_size);

// ICP Convergence criteria
        pipelines::registration::ICPConvergenceCriteria criteria;

        // Set the robust kernel function
        auto robust_kernel = std::make_shared<pipelines::registration::GMLoss>(0.1);

        // Perform point-to-plane ICP
        auto result_icp = pipelines::registration::RegistrationICP(
                *downsampled_cloud1_ptr,
                *downsampled_cloud2_ptr,
                max_correspondence_distance,  // Max correspondence distance
                pred_pose,
                pipelines::registration::TransformationEstimationPointToPlane(robust_kernel),
                criteria
        );

        return result_icp.transformation_;
    }
};

#endif //OPEN3DEXTRA_G3REG_API_H
