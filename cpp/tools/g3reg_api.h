//
// Created by qzj on 6/8/24.
//

#ifndef OPEN3DEXTRA_G3REG_API_H
#define OPEN3DEXTRA_G3REG_API_H

#include "back_end/reglib.h"
#include "robot_utils/eigen_types.h"

using VoxelKey = std::tuple<int, int, int>;

class Voxel3D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Voxel3D> Ptr;

    Voxel3D(const VoxelKey key) {
        key_ = key;
    }

    int count() {
        return indices_.size();
    }

    void sort() {
        std::sort(indices_.begin(), indices_.end(), [&](int a, int b) {
            return scores_[a] > scores_[b];
        });
        std::sort(scores_.begin(), scores_.end());
    }

    std::vector<int> getTopK(const int &k) {
        if (k > indices_.size()) {
            return indices_;
        }
        return std::vector<int>(indices_.begin(), indices_.begin() + k);
    }

    void insertPoint(const int &i, const float &score) {
        indices_.push_back(i);
        scores_.push_back(score);
    }

public:
    VoxelKey key_;
    std::vector<int> indices_;
    std::vector<float> scores_;
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
        msg << match_pairs.size() << " Matched instance pairs: \n";
        msg << corr_src_points.size() << " Correspondence points: \n";

//    稠密点匹配
        std::vector<int> indices = downsample_corr_indices(corr_src_points, corr_scores_vec);
//    合并instance中心匹配和稠密点匹配
        Eigen::MatrixX3d src_corrp(match_pairs.size() + indices.size(), 3);
        Eigen::MatrixX3d ref_corrp(match_pairs.size() + indices.size(), 3);
        for (int i = 0; i < indices.size(); i++) {
            src_corrp.row(i) = corr_src_points[indices[i]];
            src_corrp.row(i) = corr_ref_points[indices[i]];
        }

        std::vector<Eigen::Vector3d> src_points, ref_points;
//    中心匹配
        for (int i = 0; i < match_pairs.size(); i++) {
            const auto &pair = match_pairs[i];
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            src_corrp.row(i + indices.size()) = src_node->centroid;
            ref_corrp.row(i + indices.size()) = ref_node->centroid;

            // Add raw point clouds，用于验证
            src_points.insert(src_points.end(), src_node->cloud->points_.begin(), src_node->cloud->points_.end());
            ref_points.insert(ref_points.end(), ref_node->cloud->points_.begin(), ref_node->cloud->points_.end());
            msg << "(" << pair.first << "," << pair.second << ") "
                << "(" << src_node->semantic << "," << ref_node->semantic << ")\n";
        }
        std::cout << "Dense correspondence points: " << src_corrp.rows() << std::endl;

//    求解位姿矩阵
        auto src_cloud = vectorToMatrix(src_points);
        auto ref_cloud = vectorToMatrix(ref_points);
        FRGresult result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud, ref_cloud, cfg_);
        std::cout << "FRG result: " << std::endl;
        std::cout << "verify time: " << result.verify_time << std::endl;
        std::cout << "total time: " << result.total_time << std::endl;
        std::cout << "tf: \n" << result.tf << std::endl;
        std::cout << msg.str() << std::endl;

        src_cloud_ptr = std::make_shared<fmfusion::O3d_Cloud>(src_points);
        ref_cloud_ptr = std::make_shared<fmfusion::O3d_Cloud>(ref_points);
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
        for (auto voxel_pair: voxel_map) {
            auto &voxel = voxel_pair.second;
            voxel->sort();
            max_num = std::max(max_num, voxel->count());
        }
        std::cout << "Initial correspondences size: " << corr_scores_vec.size() << std::endl;

        int left = 1, right = max_num;
        int best_k = -1;
        int n = 0;
        std::vector<int> indices;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            indices.clear();
            for (auto voxel_pair: voxel_map) {
                auto &voxel = voxel_pair.second;
                std::vector<int> top_k = voxel->getTopK(mid);
                indices.insert(indices.end(), top_k.begin(), top_k.end());
            }
            n = indices.size();

            if (n < target) {
                best_k = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        std::cout << "Final correspondences size: " << indices.size() << " with at max " << best_k << " points\n";
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
};

#endif //OPEN3DEXTRA_G3REG_API_H
