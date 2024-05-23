#include "ShapeEncoder.h"

namespace fmfusion
{

    at::Tensor radius_search(at::Tensor q_points, 
                        at::Tensor s_points, 
                        at::Tensor q_lengths, 
                        at::Tensor s_lengths, 
                        float radius, 
                        int neighbor_limit)
    {
        at::Tensor neighbor_indices = radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius); // (N, max_neighbors)
        if (neighbor_limit>0){
            neighbor_indices = neighbor_indices.slice(1, 0, neighbor_limit); // (N, neighbor_limit)
        }
        return neighbor_indices;
    }

    ShapeEncoder::ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder):config(config_)
    {
        std::cout<<"Init shape encoder\n";
        std::cout<<"search radius: "<<config.init_radius<<"\n";
        std::cout<<"search voxel size: "<<config.init_voxel_size<<"\n";

    };

    void ShapeEncoder::encode(const std::vector<Eigen::Vector3d> &xyz_, const std::vector<uint32_t> &labels_,
                            const std::vector<Eigen::Vector3d> &centroids_, const std::vector<uint32_t> &nodes)
    {
        std::vector<at::Tensor> points_list;
        std::vector<at::Tensor> lengths_list;
        std::vector<at::Tensor> neighbors_list;
        std::vector<at::Tensor> subsampling_list;
        std::vector<at::Tensor> upsampling_list;
        long X = xyz_.size(); // point cloud number
        long N = centroids_.size(); // node number

        //
        std::cout<<"Prepare points cloud data dict\n";
        at::Tensor points = torch::zeros({xyz_.size(), 3}, torch::kFloat32);
        at::Tensor lengths = torch::zeros({1}, torch::kInt64);
        lengths[0] = X;

        for (int i=0;i<xyz_.size();i++)
        {
            points[i][0] = xyz_[i][0];
            points[i][1] = xyz_[i][1];
            points[i][2] = xyz_[i][2];
        }
        std::cout<<"Input points: "<<points.sizes()<<std::endl;
        std::cout<<"Input lengths: "<<X <<std::endl;

        precompute_data_stack_mode(points, lengths, points_list, lengths_list, neighbors_list, subsampling_list, upsampling_list);
    
        //
        torch::Tensor nodes_feats = torch::ones({N, 1}, torch::kFloat32);
    
    
    };

    void ShapeEncoder::precompute_data_stack_mode(at::Tensor points, at::Tensor lengths,
                                                std::vector<at::Tensor> &points_list,
                                                std::vector<at::Tensor> &lengths_list,
                                                std::vector<at::Tensor> &neighbors_list,
                                                std::vector<at::Tensor> &subsampling_list,
                                                std::vector<at::Tensor> &upsampling_list)
    {
        // std::cout<<"Precompute data stack mode\n";
        // std::vector<at::Tensor> points_list;
        // std::vector<at::Tensor> lengths_list;
        // std::vector<at::Tensor> neighbors_list;
        // std::vector<at::Tensor> subsampling_list;
        // std::vector<at::Tensor> upsampling_list;
        float voxel_size = config.voxel_size;

        points_list.push_back(points);
        lengths_list.push_back(lengths);
        CHECK_CPU(points);
        CHECK_CPU(lengths);
        CHECK_IS_FLOAT(points);
        CHECK_IS_LONG(lengths);
        CHECK_CONTIGUOUS(points);
        CHECK_CONTIGUOUS(lengths);

        // Grid subsampling
        for(int i=1;i<config.num_stages;i++)
        {
            // std::cout<<"Try to subsample\n";
            auto hidden_states = grid_subsampling(points_list.back(), lengths_list.back(), voxel_size);

            at::Tensor hidden_points = hidden_states[0];
            at::Tensor hidden_length = hidden_states[1];

            points_list.push_back(hidden_points);
            lengths_list.push_back(hidden_length);
            voxel_size *= 2;
        }

        assert(lengths_list.size() == config.num_stages);
        // std::cout<<"Point lengths: "<<lengths_list<<std::endl;
        // std::cout<<"Subsampled.\n";

        // Radius neighbors
        float search_radius = config.init_radius;
        for (int i=0;i<config.num_stages;i++){
            auto cur_points = points_list[i];
            auto cur_lengths = lengths_list[i];

            at::Tensor cur_neighbors = radius_search(cur_points, 
                                                    cur_points, 
                                                    cur_lengths, 
                                                    cur_lengths, 
                                                    search_radius,
                                                    config.neighbor_limits[i]);

            neighbors_list.push_back(cur_neighbors);
            if(i<config.num_stages-1){
                auto sub_points = points_list[i+1];
                auto sub_lengths = lengths_list[i+1];
                
                at::Tensor subsampling = radius_search(sub_points, 
                                                    cur_points, 
                                                    sub_lengths, 
                                                    cur_lengths, 
                                                    search_radius,
                                                    config.neighbor_limits[i]);
                subsampling_list.push_back(subsampling);

                at::Tensor upsampling = radius_search(cur_points, 
                                                    sub_points, 
                                                    cur_lengths, 
                                                    sub_lengths, 
                                                    search_radius,
                                                    config.neighbor_limits[i+1]);
                upsampling_list.push_back(upsampling);
            }

            search_radius *= 2;
        }

        


    };




} // namespace fmfusion
