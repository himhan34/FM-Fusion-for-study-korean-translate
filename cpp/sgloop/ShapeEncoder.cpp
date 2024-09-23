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
        if (neighbor_limit > 0)
        {
            neighbor_indices = neighbor_indices.slice(1, 0, neighbor_limit); // (N, neighbor_limit)
        }
        return neighbor_indices;
    }

    ShapeEncoder::ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder, int cuda_number) : 
        config(config_)
    {
        // std::string shape_encoder_dir = weight_folder + "/instance_shape_encoder_v1.pt";
        cuda_device_string = "cuda:" + std::to_string(cuda_number);
        encoder_v2 = torch::jit::load(weight_folder + "/instance_shape_encoder_v2.pt");
        encoder_v2.to(cuda_device_string);
        std::cout<<"Load shape encoder v2 from "<<weight_folder
                <<" on cuda:"<<cuda_number<<std::endl;
        assert(config.padding=="zero" || config.padding=="random");
        // std::cout<<config.print_msg();

    };

    void ShapeEncoder::encode(const std::vector<Eigen::Vector3d> &xyz,  
                              const std::vector<int> &length_vec, 
                              const std::vector<uint32_t> &labels,
                              const std::vector<Eigen::Vector3d> &centroids_, 
                              const std::vector<uint32_t> &nodes,
                              torch::Tensor &node_shape_feats, 
                              torch::Tensor &node_knn_points, 
                              torch::Tensor &node_knn_feats,
                              float &encoding_time,
                              std::string hidden_feat_dir)
    {
        long X = xyz.size();        // point cloud number
        long N = centroids_.size(); // node number
        open3d::utility::Timer timer;
        std::stringstream msg;

        timer.Start();
        std::vector<at::Tensor> points_list;
        std::vector<at::Tensor> lengths_list;
        std::vector<at::Tensor> neighbors_list;
        std::vector<at::Tensor> subsampling_list;
        std::vector<at::Tensor> upsampling_list;
        at::Tensor points_feats = torch::ones({X, 1}, torch::kFloat32);//.to(cuda_device_string);

        //
        float xyz_arr[X][3];
        int B = length_vec.size();
        at::Tensor lengths= torch::zeros({B}, torch::kInt64);
        if(B==1) lengths[0] = length_vec[0];
        else if(B==2) {
            float length_arr[] = {length_vec[0], length_vec[1]};
            lengths = torch::from_blob(length_arr, {B}).to(torch::kInt64);
        }
        else{
            std::cerr<<"Invalid length vector size.\n";
            assert(false);
            return;
        }

        for (int i = 0; i < X; i++)
        {
            xyz_arr[i][0] = xyz[i][0];
            xyz_arr[i][1] = xyz[i][1];
            xyz_arr[i][2] = xyz[i][2];
        }
        at::Tensor points = torch::from_blob(xyz_arr, {X, 3}).to(torch::kFloat32);
        timer.Stop();
        msg<<"concat: "
            <<std::fixed<<std::setprecision(1)
            <<timer.GetDurationInMillisecond()<<" ms, ";

        timer.Start();
        precompute_data_stack_mode(points, lengths, points_list, lengths_list, neighbors_list, subsampling_list, upsampling_list);
        timer.Stop();
        msg<<"precompute: "<<timer.GetDurationInMillisecond()<<" ms, ";

        //
        timer.Start();
        at::Tensor labels_f = torch::zeros({points_list[1].size(0)}, torch::kInt32);
        at::Tensor node_point_indices = torch::zeros({N, config.K_shape_samples}, torch::kInt32);
        at::Tensor node_knn_indices = torch::zeros({N, config.K_match_samples}, torch::kInt32);
        associate_f_points(xyz, labels, points_list[1], labels_f);
        timer.Stop();
        msg<<"associate: "<<timer.GetDurationInMillisecond()<<" ms, ";

        timer.Start();
        sample_node_f_points(labels_f, nodes, node_point_indices, config.K_shape_samples);
        sample_node_f_points(labels_f, nodes, node_knn_indices, config.K_match_samples, 1);

        
        torch::Tensor nodes_feats = torch::ones({N}, torch::kFloat32).to(cuda_device_string);
        // node_point_indices = node_point_indices.to(cuda_device_string);
        node_knn_indices = node_knn_indices.to(cuda_device_string);
        timer.Stop();
        msg<<"sampling: "<<timer.GetDurationInMillisecond()<<" ms, ";

        //
        timer.Start();
        torch::Tensor f_points_feats;
        auto output = encoder_v2({points_feats.to(cuda_device_string),
                                points_list[0].to(torch::kFloat32).to(cuda_device_string),
                                points_list[1].to(torch::kFloat32).to(cuda_device_string),
                                points_list[2].to(torch::kFloat32).to(cuda_device_string),
                                points_list[3].to(torch::kFloat32).to(cuda_device_string),
                                neighbors_list[0].to(torch::kInt64).to(cuda_device_string),
                                neighbors_list[1].to(torch::kInt64).to(cuda_device_string),
                                neighbors_list[2].to(torch::kInt64).to(cuda_device_string),
                                neighbors_list[3].to(torch::kInt64).to(cuda_device_string),
                                subsampling_list[0].to(torch::kInt64).to(cuda_device_string),
                                subsampling_list[1].to(torch::kInt64).to(cuda_device_string),
                                subsampling_list[2].to(torch::kInt64).to(cuda_device_string),
                                upsampling_list[0].to(torch::kInt64).to(cuda_device_string),
                                upsampling_list[1].to(torch::kInt64).to(cuda_device_string),
                                upsampling_list[2].to(torch::kInt64).to(cuda_device_string),
                                node_point_indices.to(cuda_device_string)}).toTuple();
        node_shape_feats = output->elements()[0].toTensor();
        f_points_feats = output->elements()[1].toTensor();
        

        assert(node_shape_feats.sizes()[0] == N);
        assert(f_points_feats.sizes()[0] == points_list[1].size()[0]);

        node_shape_feats = torch::nn::functional::normalize(node_shape_feats, 
                                                            torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        timer.Stop();
        encoding_time = timer.GetDurationInMillisecond();
        msg << "encode: " << encoding_time << " ms, ";

        //
        timer.Start();
        torch::Tensor padded_points_f = torch::cat({points_list[1], torch::zeros({1, 3}, torch::kFloat32)}, 0).to(cuda_device_string);
        torch::Tensor padded_feats_f = torch::cat({f_points_feats, torch::zeros({1, f_points_feats.size(1)}, torch::kFloat32).to(cuda_device_string)}, 0);
        node_knn_points = torch::index_select(padded_points_f, 0, node_knn_indices.view(-1)).view({N, config.K_match_samples, -1});
        node_knn_feats = torch::index_select(padded_feats_f, 0, node_knn_indices.view(-1)).view({N, config.K_match_samples, -1});
        timer.Stop();
        msg << "knn: " << timer.GetDurationInMillisecond() << " ms";
        // std::cout<<"node knn shape: "<<node_knn_feats.sizes()<<std::endl;

        std::cout<<msg.str()<<std::endl;

        if (hidden_feat_dir!="")
        {
            torch::save({points_feats, 
                        points_list[0], points_list[1], points_list[2], points_list[3],
                        neighbors_list[0], neighbors_list[1], neighbors_list[2], neighbors_list[3],
                        subsampling_list[0], subsampling_list[1], subsampling_list[2],
                        upsampling_list[0], upsampling_list[1], upsampling_list[2],
                        node_point_indices}, 
                        hidden_feat_dir);
        }

    };

    void ShapeEncoder::precompute_data_stack_mode(at::Tensor points, at::Tensor lengths,
                                                  std::vector<at::Tensor> &points_list,
                                                  std::vector<at::Tensor> &lengths_list,
                                                  std::vector<at::Tensor> &neighbors_list,
                                                  std::vector<at::Tensor> &subsampling_list,
                                                  std::vector<at::Tensor> &upsampling_list)
    {
        float voxel_size = 2 * config.init_voxel_size;

        points_list.push_back(points);
        lengths_list.push_back(lengths);
        CHECK_CPU(points);
        CHECK_CPU(lengths);
        CHECK_IS_FLOAT(points);
        CHECK_IS_LONG(lengths);
        CHECK_CONTIGUOUS(points);
        CHECK_CONTIGUOUS(lengths);

        // Grid subsampling
        for (int i = 1; i < config.num_stages; i++)
        {
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
        for (int i = 0; i < config.num_stages; i++)
        {
            auto cur_points = points_list[i];
            auto cur_lengths = lengths_list[i];

            at::Tensor cur_neighbors = radius_search(cur_points,
                                                     cur_points,
                                                     cur_lengths,
                                                     cur_lengths,
                                                     search_radius,
                                                     config.neighbor_limits[i]);

            neighbors_list.push_back(cur_neighbors);
            if (i < config.num_stages - 1)
            {
                auto sub_points = points_list[i + 1];
                auto sub_lengths = lengths_list[i + 1];

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
                                                      config.neighbor_limits[i + 1]);
                upsampling_list.push_back(upsampling);
            }

            search_radius *= 2;
        }
    };

    void ShapeEncoder::associate_f_points(const std::vector<Eigen::Vector3d> &xyz, const std::vector<uint32_t> &labels,
                                          const at::Tensor &points_f, at::Tensor &labels_f)
    {
        int Xf = points_f.size(0);
        // std::cout<<"Xf: "<<Xf<<std::endl;
        open3d::geometry::PointCloud pcd(xyz);
        open3d::geometry::KDTreeFlann kdtree(pcd);
        float SEARCH_RADIUS = 0.5;
        float labels_f_array[Xf];

        for (int i = 0; i < Xf; i++)
        {
            float x = points_f[i][0].item<float>();
            float y = points_f[i][1].item<float>();
            float z = points_f[i][2].item<float>();
            Eigen::Vector3d query_point(x, y, z);
            std::vector<int> q_indices;
            std::vector<double> q_distances;

            int result = kdtree.SearchKNN(query_point, 1, q_indices, q_distances);
            assert(result > 0);
            assert(q_distances[0] < SEARCH_RADIUS);
            // labels_f[i] = labels[q_indices[0]];
            labels_f_array[i] = labels[q_indices[0]];
        }

        labels_f = torch::from_blob(labels_f_array, {Xf}).to(torch::kInt32);

    }

    void ShapeEncoder::sample_node_f_points(const at::Tensor &labels_f, const std::vector<uint32_t> &nodes,
                                            at::Tensor &node_f_points, int K, int padding_mode)
    {
        int Nf = labels_f.size(0);
        for (const uint32_t node_id : nodes)
        {
            // Find the labels that equal to node_id
            at::Tensor mask = labels_f.eq(node_id);
            at::Tensor node_points = mask.nonzero().squeeze(1);
            if (node_points.size(0) < K)
            { // padding the small instances
                int padding = K - node_points.size(0);
                at::Tensor padding_points = torch::zeros({padding}, torch::kInt32);
                if (padding_mode == 0){
                    if (config.padding=="zero") padding_points.fill_(node_points[0]);
                    else if(config.padding=="random"){
                        torch::Tensor node_weights = torch::ones_like(node_points).to(torch::kFloat32) / node_points.size(0);
                        padding_points = node_points.index_select(0, torch::multinomial(node_weights, padding, true));
                        // padding_points = torch::multinomial(node_weights, padding, true);
                    }
                }
                else if (padding_mode == 1)
                    padding_points.fill_(Nf);
                node_points = torch::cat({node_points, padding_points}, 0);
            }
            else
            {                                                                                    // random sample K points
                node_points = node_points.index_select(0, torch::randperm(node_points.size(0))); // reorder
                node_points = node_points.slice(0, 0, K);                                        // sample K points
            }
            node_f_points[node_id] = node_points;

            // std::cout<<"Node "<<node_id<<" has "<<node_points.size(0)<<" points\n";
        }
    }

} // namespace fmfusion
