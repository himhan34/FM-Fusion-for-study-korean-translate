#ifndef SHAPE_ENCODER_H_
#define SHAPE_ENCODER_H_

#include <vector>
#include <Eigen/Core>
#include <torch/torch.h>

#include "thirdparty/extensions/cpu/grid_subsampling/grid_subsampling.h"
#include "thirdparty/extensions/cpu/radius_neighbors/radius_neighbors.h"

namespace fmfusion
{
    struct ShapeEncoderConfig
    {
        int num_stages = 4;
        float voxel_size = 0.025;
        int neighbor_limits[4] = {38, 9, 9, 9};
        float init_voxel_size = 2.5 * voxel_size;
        float init_radius = 2.0 * voxel_size;
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

        void encode(const std::vector<Eigen::Vector3d> &xyz_, const std::vector<uint32_t> &labels_, 
                    const std::vector<Eigen::Vector3d> &centroids_, const std::vector<uint32_t> &nodes);

    private:
        void precompute_data_stack_mode(at::Tensor points, at::Tensor lengths,
                                        std::vector<at::Tensor> &points_list,
                                        std::vector<at::Tensor> &lengths_list,
                                        std::vector<at::Tensor> &neighbors_list,
                                        std::vector<at::Tensor> &subsampling_list,
                                        std::vector<at::Tensor> &upsampling_list);


    
    private:
        ShapeEncoderConfig config;
        std::vector<Eigen::Vector3d> xyz;
        std::vector<uint32_t> labels;
        at::Tensor points;

    };
    
    typedef std::shared_ptr<ShapeEncoder> ShapeEncoderPtr;
} // namespace fmfusion



#endif
