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

    ShapeEncoder::ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder) : config(config_)
    {
        // std::cout << "Init shape encoder\n";
        // std::cout << "search radius: " << config.init_radius << "\n";
        // std::cout << "search voxel size: " << config.init_voxel_size << "\n";
        std::string shape_encoder_dir = weight_folder + "/instance_shape_encoder.pt";

        //
        try
        {
            encoder = torch::jit::load(shape_encoder_dir);
            encoder.to(torch::kCUDA);
            std::cout << "Load shape encoder from " << shape_encoder_dir << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    };

    void ShapeEncoder::encode(const std::vector<Eigen::Vector3d> &xyz, const std::vector<uint32_t> &labels,
                              const std::vector<Eigen::Vector3d> &centroids_, const std::vector<uint32_t> &nodes,
                              torch::Tensor &node_shape_feats, torch::Tensor &node_knn_points, torch::Tensor &node_knn_feats)
    {
        long X = xyz.size();        // point cloud number
        long N = centroids_.size(); // node number

        std::vector<at::Tensor> points_list;
        std::vector<at::Tensor> lengths_list;
        std::vector<at::Tensor> neighbors_list;
        std::vector<at::Tensor> subsampling_list;
        std::vector<at::Tensor> upsampling_list;
        at::Tensor points_feats = torch::ones({X, 1}, torch::kFloat32).to(torch::kCUDA);

        //
        float xyz_arr[X][3];
        // at::Tensor points = torch::zeros({xyz.size(), 3}, torch::kFloat32);
        at::Tensor lengths = torch::zeros({1}, torch::kInt64);
        lengths[0] = X;

        for (int i = 0; i < X; i++)
        {
            xyz_arr[i][0] = xyz[i][0];
            xyz_arr[i][1] = xyz[i][1];
            xyz_arr[i][2] = xyz[i][2];
        }
        at::Tensor points = torch::from_blob(xyz_arr, {X, 3}).to(torch::kFloat32);

        open3d::utility::Timer timer;
        timer.Start();
        precompute_data_stack_mode(points, lengths, points_list, lengths_list, neighbors_list, subsampling_list, upsampling_list);
        timer.Stop();
        // std::cout << "prepare data stack takes " << timer.GetDurationInMillisecond() << " ms\n";
        // std::cout << "Input points: " << X << ", "
        //           << "fine-level points:" << points_list[1].size(0) << "\n";
        //
        // at::Tensor points_f = points_list[1].to(torch::kFloat32).to(torch::kCUDA);
        at::Tensor labels_f = torch::zeros({points_list[1].size(0)}, torch::kInt32);
        at::Tensor node_point_indices = torch::zeros({N, config.K_shape_samples}, torch::kInt32);
        at::Tensor node_knn_indices = torch::zeros({N, config.K_match_samples}, torch::kInt32);
        associate_f_points(xyz, labels, points_list[1], labels_f);
        sample_node_f_points(labels_f, nodes, node_point_indices, config.K_shape_samples);
        sample_node_f_points(labels_f, nodes, node_knn_indices, config.K_match_samples, 1);

        //
        torch::Tensor nodes_feats = torch::ones({N}, torch::kFloat32).to(torch::kCUDA);
        node_point_indices = node_point_indices.to(torch::kCUDA);
        node_knn_indices = node_knn_indices.to(torch::kCUDA);

        //
        timer.Start();
        auto output = encoder({points_feats,
                               points_list[0].to(torch::kFloat32).to(torch::kCUDA),
                               points_list[1].to(torch::kFloat32).to(torch::kCUDA),
                               points_list[2].to(torch::kFloat32).to(torch::kCUDA),
                               points_list[3].to(torch::kFloat32).to(torch::kCUDA),
                               neighbors_list[0].to(torch::kInt64).to(torch::kCUDA),
                               neighbors_list[1].to(torch::kInt64).to(torch::kCUDA),
                               neighbors_list[2].to(torch::kInt64).to(torch::kCUDA),
                               neighbors_list[3].to(torch::kInt64).to(torch::kCUDA),
                               subsampling_list[0].to(torch::kInt64).to(torch::kCUDA),
                               subsampling_list[1].to(torch::kInt64).to(torch::kCUDA),
                               subsampling_list[2].to(torch::kInt64).to(torch::kCUDA),
                               upsampling_list[0].to(torch::kInt64).to(torch::kCUDA),
                               upsampling_list[1].to(torch::kInt64).to(torch::kCUDA),
                               upsampling_list[2].to(torch::kInt64).to(torch::kCUDA),
                               node_point_indices}).toTuple();
        node_shape_feats = output->elements()[0].toTensor();
        torch::Tensor f_points_feats = output->elements()[1].toTensor();
        timer.Stop();
        assert(node_shape_feats.sizes()[0] == N);
        assert(f_points_feats.sizes()[0] == points_list[1].size()[0]);

        // std::cout << "Run point encoder takes " << timer.GetDurationInMillisecond() << " ms\n";
        // std::cout<<points_list[1].size(0)<<","<<points_list[1].size(1)<<"\n";

        //
        torch::Tensor padded_points_f = torch::cat({points_list[1], torch::zeros({1, 3}, torch::kFloat32)}, 0).to(torch::kCUDA);
        torch::Tensor padded_feats_f = torch::cat({f_points_feats, torch::zeros({1, f_points_feats.size(1)}, torch::kFloat32).to(torch::kCUDA)}, 0);
        node_knn_points = torch::index_select(padded_points_f, 0, node_knn_indices.view(-1)).view({N, config.K_match_samples, -1});
        node_knn_feats = torch::index_select(padded_feats_f, 0, node_knn_indices.view(-1)).view({N, config.K_match_samples, -1});
        // std::cout<<"node knn shape: "<<node_knn_feats.sizes()<<std::endl;

    };

    void ShapeEncoder::precompute_data_stack_mode(at::Tensor points, at::Tensor lengths,
                                                  std::vector<at::Tensor> &points_list,
                                                  std::vector<at::Tensor> &lengths_list,
                                                  std::vector<at::Tensor> &neighbors_list,
                                                  std::vector<at::Tensor> &subsampling_list,
                                                  std::vector<at::Tensor> &upsampling_list)
    {
        float voxel_size = 2 * config.voxel_size;

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
            labels_f[i] = labels[q_indices[0]];
        }
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
            { // fill with padding value
                int padding = K - node_points.size(0);
                at::Tensor padding_points = torch::zeros({padding}, torch::kInt32);
                if (padding_mode == 0)
                    padding_points.fill_(node_points[0]);
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
