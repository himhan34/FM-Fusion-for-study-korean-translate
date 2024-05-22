#include "ShapeEncoder.h"

namespace fmfusion
{
    ShapeEncoder::ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder):config(config_)
    {
        std::cout<<"Init shape encoder\n";

    };

    void ShapeEncoder::encode(const std::vector<Eigen::Vector3d> &xyz_, const std::vector<uint32_t> &labels_)
    {
        // xyz = xyz_;
        // labels = labels_;
        std::cout<<"Encoding graph shapes\n";
        int N = xyz_.size();

        at::Tensor points = torch::zeros({xyz_.size(), 3}, torch::kFloat32);
        //  set length to N
        at::Tensor lengths = torch::zeros({1}, torch::kInt32);
        lengths[0] = N;

        for (int i=0;i<xyz_.size();i++)
        {
            points[i][0] = xyz_[i][0];
            points[i][1] = xyz_[i][1];
            points[i][2] = xyz_[i][2];
        }

        // std::cout<<points<<std::endl;
        std::cout<<"Shape tensor: "<<points.sizes()<<std::endl;
        CHECK_CPU(points);

        // precompute_data_stack_mode(points, lengths);


    };

    void ShapeEncoder::precompute_data_stack_mode(at::Tensor points, at::Tensor lengths)
    {
        std::cout<<"Precompute data stack mode\n";
        std::vector<at::Tensor> points_list;
        std::vector<at::Tensor> lengths_list;
        std::vector<at::Tensor> neighbors_list;
        std::vector<at::Tensor> subsampling_list;
        std::vector<at::Tensor> upsampling_list;


        


    };




} // namespace fmfusion
