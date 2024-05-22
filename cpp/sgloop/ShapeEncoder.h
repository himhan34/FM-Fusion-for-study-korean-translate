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
        float voxel_size = 0.05;
        int neighbor_limits[4] = {38, 36, 36, 38};
        float init_voxel_size = 0.125;
        float init_radius = 0.1;
    };

    class ShapeEncoder
    {
    /// \brief  Shape encoder for one single scene graph.
    public:
        ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder);
        ~ShapeEncoder(){};

        void encode(const std::vector<Eigen::Vector3d> &xyz_, const std::vector<uint32_t> &labels_);

    private:
        void precompute_data_stack_mode(at::Tensor points, at::Tensor lengths);


    
    private:
        ShapeEncoderConfig config;
        std::vector<Eigen::Vector3d> xyz;
        std::vector<uint32_t> labels;
        at::Tensor points;

    };
    
    typedef std::shared_ptr<ShapeEncoder> ShapeEncoderPtr;
} // namespace fmfusion



#endif
