//
// Created by qzj on 6/8/24.
//

#ifndef OPEN3DEXTRA_G3REG_API_H
#define OPEN3DEXTRA_G3REG_API_H

#include "back_end/reglib.h"
#include "mapping/Instance.h"
#include "robot_utils/eigen_types.h"
#include "utils/opt_utils.h"

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

//template<typename T>
//VoxelKey point_to_voxel_key(const Eigen::Matrix<T, 3, 1> &point, const float &voxel_size) {
//    int x = static_cast<int>(std::floor(point.x() / voxel_size));
//    int y = static_cast<int>(std::floor(point.y() / voxel_size));
//    int z = static_cast<int>(std::floor(point.z() / voxel_size));
//    return std::make_tuple(x, y, z);
//}

class G3RegAPI {
public:
    class Config : public g3reg::Config {
    public:
        double search_radius = 0.5;
        double icp_voxel = 0.2;
        double ds_voxel = 0.5;
        int ds_num = 9;
    };

public:
    Config cfg_;
    VoxelMap3D voxel_map;
    FRGresult reg_result;

    G3RegAPI(Config &cfg) : cfg_(cfg) {

    }

    void estimate_pose(const std::vector<fmfusion::NodePtr> &src_nodes,
                       const std::vector<fmfusion::NodePtr> &ref_nodes,
                       const std::vector<std::pair<uint32_t, uint32_t>> &match_pairs,
                       const std::vector<float> &corr_scores_vec,
                       const std::vector<Eigen::Vector3d> &corr_src_points,
                       const std::vector<Eigen::Vector3d> &corr_ref_points,
                       const fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                       const fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr,
                       Eigen::Matrix4d &pose) {
        // assert(src_corrp.rows() == ref_corrp.rows());
        std::stringstream msg;
        std::vector<int> indices;
        double ds_time, data_time = 0;
        robot_utils::TicToc t;
        //    稠密点匹配
        if (corr_src_points.size() > 0)
            indices = downsample_corr_indices(corr_src_points, corr_scores_vec);
        ds_time = t.toc();

        //    合并instance中心匹配和稠密点匹配
        Eigen::MatrixX3d src_corrp(match_pairs.size() + indices.size(), 3);
        Eigen::MatrixX3d ref_corrp(match_pairs.size() + indices.size(), 3);
        for (int i = 0; i < indices.size(); i++) {
            src_corrp.row(i) = corr_src_points[indices[i]];
            ref_corrp.row(i) = corr_ref_points[indices[i]];
        }
        msg << "Correspondence points: " << corr_src_points.size() << " -> " << indices.size() << "\n";

        //    中心匹配
//        msg << match_pairs.size() << " Matched instance pairs: \n";
        int src_cloud_num = 0, ref_cloud_num = 0;
        for (int i = 0; i < match_pairs.size(); i++) {
            src_cloud_num += src_nodes[match_pairs[i].first]->cloud->points_.size();
            ref_cloud_num += ref_nodes[match_pairs[i].second]->cloud->points_.size();
        }
        // src_cloud_ptr->points_.reserve(src_cloud_num);
        // src_cloud_ptr->colors_.reserve(src_cloud_num);
        // src_cloud_ptr->normals_.reserve(src_cloud_num);
        // ref_cloud_ptr->points_.reserve(ref_cloud_num);
        // ref_cloud_ptr->colors_.reserve(ref_cloud_num);
        // ref_cloud_ptr->normals_.reserve(ref_cloud_num);
        for (int i = 0; i < match_pairs.size(); i++) {
            const auto &pair = match_pairs[i];
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            src_corrp.row(i + indices.size()) = src_node->centroid;
            ref_corrp.row(i + indices.size()) = ref_node->centroid;

            // Add raw point clouds，用于验证
            // MergePointClouds(src_cloud_ptr, src_node->cloud);
            // MergePointClouds(ref_cloud_ptr, ref_node->cloud);

            // msg << "(" << pair.first << "," << pair.second << ") "
            //     << "(" << src_node->semantic << "," << ref_node->semantic << ")\n";
        }
        data_time = t.toc();
        std::cout << msg.str() << std::endl;

//    求解位姿矩阵x
        const Eigen::MatrixX3d &src_cloud_mat = vectorToMatrix(src_cloud_ptr->points_);
        const Eigen::MatrixX3d &ref_cloud_mat = vectorToMatrix(ref_cloud_ptr->points_);
        double vec3mat_time = t.toc();
        FRGresult result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud_mat, ref_cloud_mat, cfg_);
        // std::cout << "FRG result: " << std::endl;
        std::cout << src_cloud_ptr->points_.size() << " src points, "
                  << ref_cloud_ptr->points_.size() << " ref points\n";
        std::cout << "ds_time: " << ds_time << ", data_time: " << data_time
                  << ", vec3mat_time: " << vec3mat_time
                  << ", reg_time(verify): "
                  << result.total_time << "(" << result.verify_time << ")" << std::endl;
        // std::cout << "tf: \n" << result.tf << std::endl;
        pose = result.tf;
    };

    void estimate_pose(const std::vector<Eigen::Vector3d> &src_centroids,
                       const std::vector<Eigen::Vector3d> &ref_centroids,
                       const std::vector<Eigen::Vector3d> &corr_src_points,
                       const std::vector<Eigen::Vector3d> &corr_ref_points, double &inlier_ratio) {

        Eigen::MatrixX3d src_corrp(src_centroids.size() + corr_src_points.size(), 3);
        Eigen::MatrixX3d ref_corrp(ref_centroids.size() + corr_ref_points.size(), 3);
        for (int i = 0; i < corr_src_points.size(); i++) {
            src_corrp.row(i) = corr_src_points[i];
            ref_corrp.row(i) = corr_ref_points[i];
        }
        for (int i = 0; i < src_centroids.size(); i++) {
            src_corrp.row(i + corr_src_points.size()) = src_centroids[i];
            ref_corrp.row(i + corr_src_points.size()) = ref_centroids[i];
        }
        Eigen::Matrix4d T_init = gtsam::svdSE3(src_corrp.transpose(), ref_corrp.transpose());
        T_init = gtsam::gncSE3(src_corrp.transpose(), ref_corrp.transpose(), T_init);

        // Compute the inlier ratio
        double threshold = 0.5; // Define a suitable threshold
        int inlier_count = 0;
        Eigen::Matrix3d rotation = T_init.block<3, 3>(0, 0);
        Eigen::Vector3d translation = T_init.block<3, 1>(0, 3);
        for (int i = 0; i < ref_corrp.rows(); i++) {
            Eigen::Vector3d src_transformed = rotation * src_corrp.row(i).transpose() + translation;
            if ((src_transformed - ref_corrp.row(i).transpose()).norm() < threshold) {
                inlier_count++;
            }
        }
        inlier_ratio = static_cast<double>(inlier_count) / ref_corrp.rows();

        // Store the result
        reg_result.tf = T_init;
        reg_result.candidates = {T_init};
    }

    void estimate_pose(const std::vector<Eigen::Vector3d> &src_centroids,
                       const std::vector<Eigen::Vector3d> &ref_centroids,
                       const fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                       const fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr
    ) {
        // assert(src_corrp.rows() == ref_corrp.rows());
        std::stringstream msg;
        double ds_time, data_time = 0;
        robot_utils::TicToc t;
        ds_time = t.toc();

        //    合并instance中心匹配和稠密点匹配
        Eigen::MatrixX3d src_corrp(src_centroids.size(), 3);
        Eigen::MatrixX3d ref_corrp(ref_centroids.size(), 3);
        for (int i = 0; i < src_centroids.size(); i++) {
            src_corrp.row(i) = src_centroids[i];
            ref_corrp.row(i) = ref_centroids[i];
        }

//    求解位姿矩阵x
        const Eigen::MatrixX3d &src_cloud_mat = vectorToMatrix(src_cloud_ptr->points_);
        const Eigen::MatrixX3d &ref_cloud_mat = vectorToMatrix(ref_cloud_ptr->points_);
        double vec3mat_time = t.toc();
        reg_result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud_mat, ref_cloud_mat, cfg_);
        // std::cout << "FRG result: " << std::endl;
        std::cout << src_cloud_ptr->points_.size() << " src points, "
                  << ref_cloud_ptr->points_.size() << " ref points\n";
        std::cout << "ds_time: " << ds_time << ", data_time: " << data_time
                  << ", vec3mat_time: " << vec3mat_time
                  << ", reg_time(verify): "
                  << reg_result.total_time << "(" << reg_result.verify_time << ")" << std::endl;
        // std::cout << "tf: \n" << result.tf << std::endl;
    };

    void estimate_pose(const std::vector<Eigen::Vector3d> &src_centroids,
                       const std::vector<Eigen::Vector3d> &ref_centroids,
                       const std::vector<Eigen::Vector3d> &corr_src_points,
                       const std::vector<Eigen::Vector3d> &corr_ref_points,
                       const std::vector<float> &corr_scores_vec,
                       const fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                       const fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr
    ) {
        // assert(src_corrp.rows() == ref_corrp.rows());
        std::stringstream msg;
        std::vector<int> indices;
        double ds_time, data_time = 0;
        robot_utils::TicToc t;
        //    稠密点匹配
        if (corr_src_points.size() > 0) {
            indices = downsample_corr_indices(corr_src_points, corr_scores_vec);
        }
        ds_time = t.toc();

        //    合并instance中心匹配和稠密点匹配
        Eigen::MatrixX3d src_corrp(src_centroids.size() + indices.size(), 3);
        Eigen::MatrixX3d ref_corrp(ref_centroids.size() + indices.size(), 3);
        for (int i = 0; i < indices.size(); i++) {
            src_corrp.row(i) = corr_src_points[indices[i]];
            ref_corrp.row(i) = corr_ref_points[indices[i]];
        }
        for (int i = 0; i < src_centroids.size(); i++) {
            src_corrp.row(i + indices.size()) = src_centroids[i];
            ref_corrp.row(i + indices.size()) = ref_centroids[i];
        }

//    求解位姿矩阵x
        const Eigen::MatrixX3d &src_cloud_mat = vectorToMatrix(src_cloud_ptr->points_);
        const Eigen::MatrixX3d &ref_cloud_mat = vectorToMatrix(ref_cloud_ptr->points_);
        double vec3mat_time = t.toc();
        reg_result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud_mat, ref_cloud_mat, cfg_);
        // std::cout << "FRG result: " << std::endl;
        std::cout << src_cloud_ptr->points_.size() << " src points, "
                  << ref_cloud_ptr->points_.size() << " ref points\n";
        std::cout << "ds_time: " << ds_time << ", data_time: " << data_time
                  << ", vec3mat_time: " << vec3mat_time
                  << ", reg_time(verify): "
                  << reg_result.total_time << "(" << reg_result.verify_time << ")" << std::endl;
        // std::cout << "tf: \n" << result.tf << std::endl;
    };


    std::vector<int> downsample_corr_indices(const std::vector<Eigen::Vector3d> &corr_src_points,
                                             const std::vector<float> &corr_scores_vec) {
        std::cout << "downsample using voxel of " << cfg_.ds_voxel << " and " << cfg_.ds_num << " points\n";
        cutCloud(corr_src_points, corr_scores_vec, cfg_.ds_voxel, voxel_map);

        int max_num = 0;
        for (const auto &voxel_pair: voxel_map) {
            auto &voxel = voxel_pair.second;
            voxel->sort();
            max_num = std::max(max_num, voxel->count());
        }

        std::vector<int> indices;
        indices.reserve(corr_src_points.size());
        for (const auto &voxel_pair: voxel_map) {
            auto &voxel = voxel_pair.second;
            const auto &top_k = voxel->getTopK(cfg_.ds_num);
            for (size_t i = 0; i < top_k.size(); ++i) {
                indices.push_back(top_k[i].second);
            }
        }

        std::cout << "indices: " << indices.size() << std::endl;
        if (indices.size() > 1000) {
            int left = 1, right = cfg_.ds_num;
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

                if (indices.size() < 1000) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            std::cout << "downsample to " << indices.size() << " points\n";
        }
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

    void analyze_correspondences(const std::vector<Eigen::Vector3d> &src_centroids,
                                 const std::vector<Eigen::Vector3d> &ref_centroids,
                                 const std::vector<Eigen::Vector3d> &corr_src_points,
                                 const std::vector<Eigen::Vector3d> &corr_ref_points,
                                 const std::map<int, std::vector<int>> &ins_corr_map,
                                 const Eigen::Matrix4d &gt_pose) {
        Eigen::Matrix3d R = gt_pose.block<3, 3>(0, 0);
        Eigen::Vector3d t = gt_pose.block<3, 1>(0, 3);
        double total_num = 0, inlier_num = 0;
        total_num = src_centroids.size();
        for (int i = 0; i < total_num; i++) {
            double error = (R * src_centroids[i] + t - ref_centroids[i]).norm();
            if (error < 0.5) {
                inlier_num++;
            }
        }
        std::cout << "Instance level: " << "inlier num: " << inlier_num << ", total num: " << total_num << ", ratio: "
                  << inlier_num / total_num << std::endl;

        for (const auto &pair: ins_corr_map) {
            std::vector<int> corr_indices = pair.second;
            total_num = corr_indices.size();
            inlier_num = 0;
            for (int i = 0; i < total_num; i++) {
                double error = (R * corr_src_points[corr_indices[i]] + t - corr_ref_points[corr_indices[i]]).norm();
                if (error < 0.5) {
                    inlier_num++;
                }
            }
            std::cout << "Dense level: " << "inlier num: " << inlier_num << ", total num: " << total_num << ", ratio: "
                      << inlier_num / total_num << std::endl;
        }
    }

    // Function to merge cloud2_ptr into cloud1_ptr
    void MergePointClouds(const fmfusion::O3d_Cloud_Ptr &cloud1_ptr, const fmfusion::O3d_Cloud_Ptr &cloud2_ptr) {

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
               Eigen::Matrix4d pred_pose = Eigen::Matrix4d::Identity()) {
        using namespace open3d;
        if (!src_cloud_ptr->HasPoints() || !ref_cloud_ptr->HasPoints()) {
            std::cout << "[WARNNING] Empty cloud. Skip ICP refine!\n";
            return pred_pose;
        }
        if (!src_cloud_ptr->HasNormals() || !ref_cloud_ptr->HasNormals()) {
            std::cout << "[WARNNING] No normals. Skip ICP refine!\n";
            return pred_pose;
        }

//        ref_cloud_ptr->EstimateNormals(geometry::KDTreeSearchParamHybrid(0.5, 30));
        auto downsampled_cloud1_ptr = src_cloud_ptr->VoxelDownSample(cfg_.icp_voxel);
//        auto downsampled_cloud2_ptr = ref_cloud_ptr->VoxelDownSample(cfg_.icp_voxel);
        auto downsampled_cloud2_ptr = ref_cloud_ptr->VoxelDownSample(0.1);

// ICP Convergence criteria
        pipelines::registration::ICPConvergenceCriteria criteria;

        // Set the robust kernel function
        auto robust_kernel = std::make_shared<pipelines::registration::GMLoss>(0.1);
//        auto robust_kernel = std::make_shared<pipelines::registration::HuberLoss>(0.2);

        // Perform point-to-plane ICP
        auto result_icp = pipelines::registration::RegistrationICP(
                *downsampled_cloud1_ptr,
                *downsampled_cloud2_ptr,
                cfg_.search_radius,  // Max correspondence distance
                pred_pose,
                pipelines::registration::TransformationEstimationPointToPlane(robust_kernel),
                criteria
        );

        return result_icp.transformation_;
    }
};

#endif //OPEN3DEXTRA_G3REG_API_H
