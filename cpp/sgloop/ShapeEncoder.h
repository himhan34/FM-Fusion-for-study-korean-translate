#ifndef SHAPE_ENCODER_H_
#define SHAPE_ENCODER_H_

#include <vector>
#include <Eigen/Core>
#include <torch/torch.h>
#include <torch/script.h> 
#include <open3d/Open3D.h>

#include "thirdparty/extensions/cpu/grid_subsampling.h"
#include "thirdparty/extensions/cpu/radius_neighbors.h"

namespace fmfusion
{
    struct ShapeEncoderConfig
    {
        int num_stages = 4;
        float voxel_size = 0.05;
        int neighbor_limits[4] = {32, 9, 9, 9};
        float init_voxel_size = 2.5 * voxel_size;
        float init_radius = 2.0 * voxel_size;
        int K_shape_samples = 1024;
        int K_match_samples = 512;
    };

    at::Tensor radius_search(at::Tensor q_points, 
                        at::Tensor s_points, 
                        at::Tensor q_lengths, 
                        at::Tensor s_lengths, 
                        float radius, 
                        int neighbor_limit);

    class ShapeEncoder
    {
    /// \brief  Shape encoder for one single scene graph.
    public:
        ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder);
        ~ShapeEncoder(){};

        /// \brief  Encode the shape of the scene graph.
        /// \param  xyz_        (X,3), The point cloud of the scene graph.
        /// \param  labels_     (X,), The node indices of each point.
        /// \param  centroids_  (N,3), The centroid of each node.
        /// \param  nodes       (N,), The index of each node.
        void encode(const std::vector<Eigen::Vector3d> &xyz, const std::vector<uint32_t> &labels, 
                    const std::vector<Eigen::Vector3d> &centroids_, const std::vector<uint32_t> &nodes,
                    torch::Tensor &node_shape_feats, 
                    torch::Tensor &node_knn_points,
                    torch::Tensor &node_knn_feats);

    private:
        void precompute_data_stack_mode(at::Tensor points, at::Tensor lengths,
                                        std::vector<at::Tensor> &points_list,
                                        std::vector<at::Tensor> &lengths_list,
                                        std::vector<at::Tensor> &neighbors_list,
                                        std::vector<at::Tensor> &subsampling_list,
                                        std::vector<at::Tensor> &upsampling_list);

        void associate_f_points(const std::vector<Eigen::Vector3d> &xyz, const std::vector<uint32_t> &labels, 
                    const at::Tensor &points_f, at::Tensor &labels_f);

        void sample_node_f_points(const at::Tensor &labels_f, const std::vector<uint32_t> &nodes,
                                        at::Tensor &node_f_points, int K=512, int padding_mode=0);
    
    private:
        ShapeEncoderConfig config;
        // std::vector<Eigen::Vector3d> xyz;
        // std::vector<uint32_t> labels;
        // at::Tensor points;
        torch::jit::script::Module encoder;

    };
    
    typedef std::shared_ptr<ShapeEncoder> ShapeEncoderPtr;
} // namespace fmfusion



#endif
