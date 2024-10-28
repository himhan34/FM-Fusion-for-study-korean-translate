//
// Created by qzj on 6/8/24.
//

#ifndef OPEN3DEXTRA_G3REG_API_H
#define OPEN3DEXTRA_G3REG_API_H

#include "back_end/reglib.h"
#include "mapping/Instance.h"
#include "robot_utils/eigen_types.h"
#include "utils/opt_utils.h"
#include "back_end/pagor/pagor.h"

using VoxelKey = std::tuple<int, int, int>;

class Voxel3D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Voxel3D> Ptr;

    explicit Voxel3D(const VoxelKey key) : key_(key) {}

    // Returns the number of score-index pairs
    int count() const {
        return score_index_tuples_.size();
    }

    // Sorts the score-index pairs in descending order of scores
    void sort() {
        std::sort(score_index_tuples_.begin(), score_index_tuples_.end(),
                  [](const std::tuple<float, int, Eigen::Vector3d> &a,
                     const std::tuple<float, int, Eigen::Vector3d> &b) {
                      return std::get<0>(a) > std::get<0>(b);
                  });
    }

    // Returns the top K score-index pairs
    std::vector<std::tuple<float, int, Eigen::Vector3d >> getTopK(int k) const {
        if (k >= score_index_tuples_.size()) {
            return score_index_tuples_;
        }
        return std::vector<std::tuple<float, int, Eigen::Vector3d >>(score_index_tuples_.begin(),
                                                                     score_index_tuples_.begin() + k);
    }

    // Returns the top K score-index pairs
    std::vector<std::tuple<float, int, Eigen::Vector3d >> getNMS(double distThd) const {
        std::vector<std::tuple<float, int, Eigen::Vector3d>> selectedTuples;
        for (const auto &tuple: score_index_tuples_) {
            bool keep = true;
            const auto &confidence = std::get<0>(tuple);
            const auto &point = std::get<2>(tuple);
            // Check if the current point is too close to any of the previously selected points
            for (const auto &selected: selectedTuples) {
                if ((point - std::get<2>(selected)).norm() < distThd) {
                    // Keep the one with higher confidence
                    if (confidence < std::get<0>(selected)) {
                        keep = false;
                        break;
                    } else {
                        // The current point has higher confidence, so remove the previous one
                        selectedTuples.erase(std::find(selectedTuples.begin(), selectedTuples.end(), selected));
                    }
                }
            }

            if (keep) {
                selectedTuples.push_back(tuple);
            }
        }

        return selectedTuples;
    }

    void insertPoint(int i, float score, const Eigen::Vector3d &point) {
        score_index_tuples_.emplace_back(score, i, point);
    }

private:
    VoxelKey key_;
    std::vector<std::tuple<float, int, Eigen::Vector3d >> score_index_tuples_;
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
        int ds_num = 1;
        int max_corr_num = 1000;
        double nms_thd = 0.05;
    };

public:
    Config cfg_;
    VoxelMap3D voxel_map;
    FRGresult reg_result;

    G3RegAPI(Config &cfg) : cfg_(cfg) {

    }

    void construct_corrp(const std::vector<Eigen::Vector3d> &src_centroids,
                         const std::vector<Eigen::Vector3d> &ref_centroids,
                         const std::vector<Eigen::Vector3d> &corr_src_points,
                         const std::vector<Eigen::Vector3d> &corr_ref_points,
                         const std::vector<float> &corr_scores_vec, std::string corr_type,
                         Eigen::MatrixX3d &src_corrp, Eigen::MatrixX3d &ref_corrp) {

        //    稠密点匹配
        std::vector<int> indices;
        if (corr_type == "none") {
            for (int i = 0; i < corr_src_points.size(); i++) {
                indices.push_back(i);
            }
        } else if (corr_type == "nms") {
//            indices = downsample_corr_nms(corr_src_points, corr_scores_vec, cfg_.nms_thd);
            for (int i = 0; i < corr_src_points.size(); i++) {
                indices.push_back(i);
            }
        } else if (corr_type == "topk") {
            indices = downsample_corr_topk(corr_src_points, corr_scores_vec);
        } else {
            std::cout << "corr_type not supported" << std::endl;
        }
        //    合并instance中心匹配和稠密点匹配
        src_corrp.resize(src_centroids.size() + indices.size(), 3);
        ref_corrp.resize(ref_centroids.size() + indices.size(), 3);
        for (int i = 0; i < indices.size(); i++) {
            src_corrp.row(i) = corr_src_points[indices[i]];
            ref_corrp.row(i) = corr_ref_points[indices[i]];
        }
        for (int i = 0; i < src_centroids.size(); i++) {
            src_corrp.row(i + indices.size()) = src_centroids[i];
            ref_corrp.row(i + indices.size()) = ref_centroids[i];
        }
    }

    void estimate_pose(const std::vector<Eigen::Vector3d> &src_centroids,
                       const std::vector<Eigen::Vector3d> &ref_centroids,
                       const std::vector<Eigen::Vector3d> &corr_src_points,
                       const std::vector<Eigen::Vector3d> &corr_ref_points,
                       const std::vector<float> &corr_scores_vec,
                       const fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                       const fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr
    ) {

        cfg_.vertex_info.type = clique_solver::VertexType::POINT;
        cfg_.tf_solver = "gnc"; //"quatro"; // gnc quatro

        std::stringstream msg;
        double data_time = 0;
        robot_utils::TicToc t;
        Eigen::MatrixX3d src_corrp, ref_corrp;
        construct_corrp(src_centroids, ref_centroids, corr_src_points, corr_ref_points, corr_scores_vec, "nms",
                        src_corrp, ref_corrp);

//    求解位姿矩阵x
        const Eigen::MatrixX3d &src_cloud_mat = vectorToMatrix(src_cloud_ptr->points_);
        const Eigen::MatrixX3d &ref_cloud_mat = vectorToMatrix(ref_cloud_ptr->points_);
        double vec3mat_time = t.toc();
        reg_result = g3reg::SolveFromCorresp(src_corrp, ref_corrp, src_cloud_mat, ref_cloud_mat, cfg_);
        // std::cout << "FRG result: " << std::endl;
        std::cout << src_cloud_ptr->points_.size() << " src points, "
                  << ref_cloud_ptr->points_.size() << " ref points\n";

        std::cout << "Time ds: data: " << data_time << ", vec3mat: " << vec3mat_time << ", feature: "
                  << reg_result.feature_time << ", clique: " << reg_result.clique_time << ", graph: "
                  << reg_result.graph_time << ", tf_solver: " << reg_result.tf_solver_time << ", verify: "
                  << reg_result.verify_time << ", total: " << reg_result.total_time << std::endl;

        // std::cout << "tf: \n" << result.tf << std::endl;
    };

    void estimate_pose_by_gems(const std::vector<fmfusion::InstancePtr> &src_instances,
                               const std::vector<fmfusion::InstancePtr> &ref_instances,
                               const fmfusion::O3d_Cloud_Ptr &src_cloud_ptr,
                               const fmfusion::O3d_Cloud_Ptr &ref_cloud_ptr) {
        std::vector<clique_solver::GraphVertex::Ptr> src_nodes, ref_nodes;
        cfg_.tf_solver = "gmm_tls"; //"quatro"; // gnc quatro gmm_tls
        cfg_.vertex_info.type = clique_solver::VertexType::ELLIPSE;
        cfg_.vertex_info.noise_bound_vec = {0.115, 0.352, 0.584, 1.005};
        for (int i = 0; i < src_instances.size(); i++) {
            if (src_instances[i] == nullptr && ref_instances[i] == nullptr) {
                std::cout << "src_instances[" << i << "] == nullptr && ref_instances[" << i << "] == nullptr"
                          << std::endl;
                continue;
            }
            if (src_instances[i] == nullptr) {
                std::cout << "src_instances[" << i << "] == nullptr" << std::endl;
            }
            if (ref_instances[i] == nullptr) {
                std::cout << "ref_instances[" << i << "] == nullptr" << std::endl;
            }
            g3reg::ClusterFeature::Ptr src_cluster_feature(
                    new g3reg::ClusterFeature(o3d2pcl(src_instances[i]->point_cloud)));
            src_nodes.push_back(src_cluster_feature->vertex());

            g3reg::ClusterFeature::Ptr ref_cluster_feature(
                    new g3reg::ClusterFeature(o3d2pcl(ref_instances[i]->point_cloud)));
            ref_nodes.push_back(ref_cluster_feature->vertex());
        }

        auto num_corresp = src_nodes.size();
        clique_solver::Association A = clique_solver::Association::Zero(num_corresp, 2);
        for (int i = 0; i < num_corresp; i++) {
            A(i, 0) = i;
            A(i, 1) = i;
        }

        g3reg::EllipsoidMatcher matcher(o3d2pcl(src_cloud_ptr), o3d2pcl(ref_cloud_ptr));
        pagor::solve(src_nodes, ref_nodes, A, matcher, reg_result);

        std::cout << src_cloud_ptr->points_.size() << " src points, "
                  << ref_cloud_ptr->points_.size() << " ref points\n";
        std::cout << "Reg_time(verify): "
                  << reg_result.total_time << "(" << reg_result.verify_time << ")" << std::endl;
        // std::cout << "tf: \n" << result.tf << std::endl;
    };

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
            indices = downsample_corr_topk(corr_src_points, corr_scores_vec);
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

    double estimate_pose_gnc(const std::vector<Eigen::Vector3d> &src_centroids,
                             const std::vector<Eigen::Vector3d> &ref_centroids,
                             const std::vector<Eigen::Vector3d> &corr_src_points,
                             const std::vector<Eigen::Vector3d> &corr_ref_points,
                             const std::vector<float> &corr_scores_vec) {
        cfg_.tf_solver = "quatro"; //"quatro"; // gnc quatro
        double inlier_ratio;
        Eigen::MatrixX3d src_corrp, ref_corrp;
        construct_corrp(src_centroids, ref_centroids, corr_src_points, corr_ref_points, corr_scores_vec, "nms",
                        src_corrp, ref_corrp);

        Eigen::Matrix4d T = solveSE3byGNC(src_corrp.transpose(), ref_corrp.transpose());

        // Compute the inlier ratio
        double threshold = 0.5; // Define a suitable threshold
        int inlier_count = 0;
        Eigen::Matrix3d rotation = T.block<3, 3>(0, 0);
        Eigen::Vector3d translation = T.block<3, 1>(0, 3);
        for (int i = 0; i < ref_corrp.rows(); i++) {
            Eigen::Vector3d src_transformed = rotation * src_corrp.row(i).transpose() + translation;
            if ((src_transformed - ref_corrp.row(i).transpose()).norm() < threshold) {
                inlier_count++;
            }
        }
        inlier_ratio = static_cast<double>(inlier_count) / ref_corrp.rows();

        // Store the result
        reg_result.tf = T;
        reg_result.candidates = {T};
        return inlier_ratio;
    }

    double compute_true_inlier_ratio(const std::vector<Eigen::Vector3d> &src_centroids,
                                     const std::vector<Eigen::Vector3d> &ref_centroids,
                                     const std::vector<Eigen::Vector3d> &corr_src_points,
                                     const std::vector<Eigen::Vector3d> &corr_ref_points, Eigen::Matrix4d &gt_tf) {

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

        // Compute the inlier ratio

        Eigen::Matrix3d rotation = gt_tf.block<3, 3>(0, 0);
        Eigen::Vector3d translation = gt_tf.block<3, 1>(0, 3);
        std::vector<double> threshold_vec = {0.2, 0.5, 1.0};
        double inlier_ratio = 0;
        std::cout << "Print the true inlier ratio: \n";
        for (int i = 0; i < threshold_vec.size(); i++) {
            double threshold = threshold_vec[i];
            int inlier_count = 0;
            for (int j = 0; j < ref_corrp.rows(); j++) {
                Eigen::Vector3d src_transformed = rotation * src_corrp.row(j).transpose() + translation;
                if ((src_transformed - ref_corrp.row(j).transpose()).norm() < threshold) {
                    inlier_count++;
                }
            }
            inlier_ratio = static_cast<double>(inlier_count) / ref_corrp.rows();
            std::cout << "threshold: " << threshold << ", true inlier ratio: " << inlier_ratio << std::endl;
        }
        return inlier_ratio;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr o3d2pcl(const fmfusion::O3d_Cloud_Ptr &o3d_cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (auto &point: o3d_cloud->points_) {
            pcl_cloud->points.emplace_back(point[0], point[1], point[2]);
        }
        return pcl_cloud;
    }

    std::vector<int> downsample_corr_nms(const std::vector<Eigen::Vector3d> &corr_src_points,
                                         const std::vector<float> &corr_scores_vec, double distThd) {
        std::vector<int> indices;

        // Mapping from sorted scores back to original indices.
        std::vector<int> score_indices(corr_scores_vec.size());
        std::iota(score_indices.begin(), score_indices.end(), 0);
        std::sort(score_indices.begin(), score_indices.end(),
                  [&corr_scores_vec](int i1, int i2) { return corr_scores_vec[i1] > corr_scores_vec[i2]; });

        // Use NMS to downsample
        for (int idx: score_indices) {
            bool add = true;
            for (size_t j = 0; j < indices.size() && add; ++j) {
                double dist = (corr_src_points[idx] - corr_src_points[indices[j]]).norm();
                if (dist < distThd) {
                    add = false;
                }
            }
            if (add) {
                indices.push_back(idx);
            }
        }

        return indices;
    }


    std::vector<int> downsample_corr_topk(const std::vector<Eigen::Vector3d> &corr_src_points,
                                          const std::vector<float> &corr_scores_vec) {
//        std::cout << "downsample using voxel of " << cfg_.ds_voxel << " and " << cfg_.ds_num << " points\n";
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
                indices.push_back(std::get<1>(top_k[i]));
            }
        }

        std::cout << "indices: " << indices.size() << std::endl;
        // double max_corr_num = 1000;
        double max_corr_num = cfg_.max_corr_num;
        if (indices.size() > max_corr_num) {
            int left = 1, right = cfg_.ds_num;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                indices.clear();
                for (const auto &voxel_pair: voxel_map) {
                    auto &voxel = voxel_pair.second;
                    const auto &top_k = voxel->getTopK(mid);
                    for (size_t i = 0; i < top_k.size(); ++i) {
                        indices.push_back(std::get<1>(top_k[i]));
                    }
                }

                if (indices.size() < max_corr_num) {
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
                voxel_iter->second->insertPoint(i, corr_scores_vec[i], corr_src_points[i]);
            } else {
                Voxel3D::Ptr voxel = Voxel3D::Ptr(new Voxel3D(position));
                voxel->insertPoint(i, corr_scores_vec[i], corr_src_points[i]);
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
        auto downsampled_cloud2_ptr = ref_cloud_ptr->VoxelDownSample(cfg_.icp_voxel);
        // auto downsampled_cloud2_ptr = ref_cloud_ptr->VoxelDownSample(0.1);

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

    Eigen::Matrix4d svdSE3(const Eigen::Matrix3Xd &src, const Eigen::Matrix3Xd &tgt,
                           const Eigen::Matrix<double, 1, Eigen::Dynamic> &W) {
        Eigen::Vector3d src_mean = src.rowwise().mean();
        Eigen::Vector3d tgt_mean = tgt.rowwise().mean();
        const Eigen::Matrix3Xd &src_centered = src - src_mean.replicate(1, src.cols());
        const Eigen::Matrix3Xd &tgt_centered = tgt - tgt_mean.replicate(1, tgt.cols());
        Eigen::MatrixXd H = src_centered * W.asDiagonal() * tgt_centered.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd R_ = svd.matrixV() * svd.matrixU().transpose();
        if (R_.determinant() < 0) {
            Eigen::MatrixXd V = svd.matrixV();
            V.col(2) *= -1;
            R_ = V * svd.matrixU().transpose();
        }
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block(0, 0, 3, 3) = R_;
        T.block(0, 3, 3, 1) = tgt_mean - R_ * src_mean;
        return T;
    }

    Eigen::Matrix4d solveSE3byGNC(const Eigen::Matrix3Xd &src, const Eigen::Matrix3Xd &dst) {
        //GNC params
        double gnc_factor = 1.4;
        double noise_bound = 0.5;
        double cost_threshold = 1e-6;
        int max_iterations = 100;

        // assert(rotation);                 // make sure R is not a nullptr
        assert(src.cols() == dst.cols()); // check dimensions of input data
        assert(gnc_factor > 1);   // make sure mu will increase        gnc_factor -> rotation_gnc_factor
        // assert(params_.noise_bound != 0); // make sure noise sigma is not zero

        // Prepare some variables
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        size_t match_size = src.cols(); // number of correspondences

        double mu = 1; // arbitrary starting mu

        double prev_cost = std::numeric_limits<double>::infinity();
        double cost = std::numeric_limits<double>::infinity();
        static double noise_bound_sq = std::pow(noise_bound, 2);
        if (noise_bound_sq < 1e-16) {
            noise_bound_sq = 1e-2;
        }
        // TEASER_DEBUG_INFO_MSG("GNC rotation estimation noise bound:" << params_.noise_bound);
        TEASER_DEBUG_INFO_MSG("GNC rotation estimation noise bound squared:" << noise_bound_sq);

        Eigen::Matrix<double, 3, Eigen::Dynamic> diffs(3, match_size);
        Eigen::Matrix<double, 1, Eigen::Dynamic> weights(1, match_size);
        weights.setOnes(1, match_size);
        Eigen::Matrix<double, 1, Eigen::Dynamic> residuals_sq(1, match_size);

        // Loop for performing GNC-TLS
        for (size_t i = 0; i < max_iterations; ++i) {   //max_iterations  = >rotation_max_iterations

            // Fix weights and perform SVD rotation estimation
            T = svdSE3(src, dst, weights);

            // Calculate residuals squared
            //diffs = (dst - (*rotation) * src).array().square();
            diffs = (dst - T.block(0, 0, 3, 3) * src - T.block(0, 3, 3, 1).replicate(1, match_size)).array().square();
            residuals_sq = diffs.colwise().sum();
            if (i == 0) {
                // Initialize rule for mu
                double max_residual = residuals_sq.maxCoeff();
                mu = 1 / (2 * max_residual / noise_bound_sq - 1);
                // Degenerate case: mu = -1 because max_residual is very small
                // i.e., little to none noise
                if (mu <= 0) {
                    TEASER_DEBUG_INFO_MSG(
                            "GNC-TLS terminated because maximum residual at initialization is very small.");
                    break;
                }
            }
            // Fix R and solve for weights in closed form
            double th1 = (mu + 1) / mu * noise_bound_sq;
            double th2 = mu / (mu + 1) * noise_bound_sq;
            cost = 0;
            for (size_t j = 0; j < match_size; ++j) {
                // Also calculate cost in this loop
                // Note: the cost calculated is using the previously solved weights
                cost += weights(j) * residuals_sq(j);

                if (residuals_sq(j) >= th1) {
                    weights(j) = 0;
                } else if (residuals_sq(j) <= th2) {
                    weights(j) = 1;
                } else {
                    weights(j) = sqrt(noise_bound_sq * mu * (mu + 1) / residuals_sq(j)) - mu;
                    assert(weights(j) >= 0 && weights(j) <= 1);
                }
            }
            // Calculate cost
            double cost_diff = std::abs(cost - prev_cost);

            // Increase mu
            mu = mu * gnc_factor;   //gnc_factor -> rotation_gnc_factor
            prev_cost = cost;

            if (cost_diff < cost_threshold) {
                TEASER_DEBUG_INFO_MSG("GNC-TLS solver terminated due to cost convergence.");
                TEASER_DEBUG_INFO_MSG("Cost diff: " << cost_diff);
                TEASER_DEBUG_INFO_MSG("Iterations: " << i);
                break;
            }
        }
        return T;
    }

};

inline void
downsample_corr_nms(std::vector<Eigen::Vector3d> &corr_src_points, std::vector<Eigen::Vector3d> &corr_tgt_points,
                    std::vector<float> &corr_scores_vec, double distThd) {
    std::vector<int> indices;

    // Mapping from sorted scores back to original indices.
    std::vector<int> score_indices(corr_scores_vec.size());
    std::iota(score_indices.begin(), score_indices.end(), 0);
    std::sort(score_indices.begin(), score_indices.end(),
              [&corr_scores_vec](int i1, int i2) { return corr_scores_vec[i1] > corr_scores_vec[i2]; });

    // Use NMS to downsample
    for (int idx: score_indices) {
        bool add = true;
        for (size_t j = 0; j < indices.size() && add; ++j) {
            double dist = (corr_src_points[idx] - corr_src_points[indices[j]]).norm();
            if (dist < distThd) {
                add = false;
            }
        }
        if (add) {
            indices.push_back(idx);
        }
    }

    // change corr_src_points and corr_tgt_points
    int original_size = corr_src_points.size();
    for (int i = 0; i < indices.size(); ++i) {
        corr_src_points.push_back(corr_src_points[indices[i]]);
        corr_tgt_points.push_back(corr_tgt_points[indices[i]]);
        corr_scores_vec.push_back(corr_scores_vec[indices[i]]);
    }
    // delete the front elements
    corr_src_points.erase(corr_src_points.begin(), corr_src_points.begin() + original_size);
    corr_tgt_points.erase(corr_tgt_points.begin(), corr_tgt_points.begin() + original_size);
    corr_scores_vec.erase(corr_scores_vec.begin(), corr_scores_vec.begin() + original_size);
}

inline void
downsample_corr_topk(std::vector<Eigen::Vector3d> &corr_src_points, std::vector<Eigen::Vector3d> &corr_tgt_points,
                     std::vector<float> &corr_scores_vec, double ds_voxel, double max_corr_num) {
    VoxelMap3D voxel_map;

    for (size_t i = 0; i < corr_src_points.size(); i++) {
        VoxelKey position = point_to_voxel_key(corr_src_points[i], ds_voxel);
        VoxelMap3D::iterator voxel_iter = voxel_map.find(position);
        if (voxel_iter != voxel_map.end()) {
            voxel_iter->second->insertPoint(i, corr_scores_vec[i], corr_src_points[i]);
        } else {
            Voxel3D::Ptr voxel = Voxel3D::Ptr(new Voxel3D(position));
            voxel->insertPoint(i, corr_scores_vec[i], corr_src_points[i]);
            voxel_map.insert(std::make_pair(position, voxel));
        }
    }

    int max_num = 0;
    for (const auto &voxel_pair: voxel_map) {
        auto &voxel = voxel_pair.second;
        voxel->sort();
        max_num = std::max(max_num, voxel->count());
    }

    std::vector<int> indices;
    indices.reserve(corr_src_points.size());
    int ds_num = 9;
    for (const auto &voxel_pair: voxel_map) {
        auto &voxel = voxel_pair.second;
        const auto &top_k = voxel->getTopK(ds_num);
        for (size_t i = 0; i < top_k.size(); ++i) {
            indices.push_back(std::get<1>(top_k[i]));
        }
    }

    if (indices.size() > max_corr_num) {
        int left = 1, right = ds_num;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            indices.clear();
            for (const auto &voxel_pair: voxel_map) {
                auto &voxel = voxel_pair.second;
                const auto &top_k = voxel->getTopK(mid);
                for (size_t i = 0; i < top_k.size(); ++i) {
                    indices.push_back(std::get<1>(top_k[i]));
                }
            }

            if (indices.size() < max_corr_num) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    int original_size = corr_src_points.size();
    for (int i = 0; i < indices.size(); ++i) {
        corr_src_points.push_back(corr_src_points[indices[i]]);
        corr_tgt_points.push_back(corr_tgt_points[indices[i]]);
        corr_scores_vec.push_back(corr_scores_vec[indices[i]]);
    }
    corr_src_points.erase(corr_src_points.begin(), corr_src_points.begin() + original_size);
    corr_tgt_points.erase(corr_tgt_points.begin(), corr_tgt_points.begin() + original_size);
    corr_scores_vec.erase(corr_scores_vec.begin(), corr_scores_vec.begin() + original_size);

}

#endif //OPEN3DEXTRA_G3REG_API_H
