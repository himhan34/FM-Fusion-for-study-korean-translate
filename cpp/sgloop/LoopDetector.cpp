#include "LoopDetector.h"


namespace fmfusion
{
    LoopDetector::LoopDetector(LoopDetectorConfig &lcd_config,
                                ShapeEncoderConfig &shape_encoder_config, 
                                SgNetConfig &sgnet_config, const std::string weight_folder):weight_folder_dir(weight_folder)
    {
        shape_encoder = std::make_shared<ShapeEncoder>(shape_encoder_config, weight_folder);
        sgnet = std::make_shared<SgNet>(sgnet_config, weight_folder);
        config = lcd_config;
        ref_sg_timestamp = -1.0;
    }

    bool LoopDetector::encode_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict, ImplicitGraph &graph_features)
    {
        open3d::utility::Timer timer;
        timer.Start();
        sgnet->graph_encoder(nodes, graph_features.node_features);
        timer.Stop();
        std::cout<<"Graph encoder takes "<<std::fixed<<std::setprecision(3)<<timer.GetDurationInMillisecond()<<" ms\n";
        if(!data_dict.nodes.empty()){ // Encode single graph shapes
            shape_encoder->encode(data_dict.xyz,data_dict.length_vec,data_dict.labels,data_dict.centroids,data_dict.nodes,
                                    graph_features.shape_features, graph_features.node_knn_points, graph_features.node_knn_features);
            graph_features.shape_embedded = true;        
            if(config.fuse_shape){
                graph_features.node_features = torch::cat({graph_features.node_features, graph_features.shape_features}, 1);
            }
            std::cout<<"Encode single graph shapes\n";
        }

        return true;
    }

    bool LoopDetector::encode_ref_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict)
    {
        return encode_scene_graph(nodes, DataDict {}, ref_features);
    }

    bool LoopDetector::encode_src_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict)
    {
        sgnet->load_bert(weight_folder_dir);
        bool ret = encode_scene_graph(nodes, DataDict {}, src_features);
        return ret;
    }

    bool LoopDetector::subscribe_ref_coarse_features(const float &latest_timestamp, 
                                                    const std::vector<std::vector<float>> &coarse_features_vec,
                                                    torch::Tensor coarse_features)
    {
        if(latest_timestamp - ref_sg_timestamp > 0.01){
            int N = coarse_features_vec.size();
            int D = coarse_features_vec[0].size();
            std::cout<<"Update subscribed node features "<<N <<" x "<<D<<"\n";
            float features_array[N][D];
            for(int i=0;i<N;i++){
                std::copy(coarse_features_vec[i].begin(), coarse_features_vec[i].end(), features_array[i]);
            }
            ref_features.node_features = torch::from_blob(features_array, {N,D}).to(torch::kFloat32).to(torch::kCUDA);
            ref_sg_timestamp = latest_timestamp;

            assert(torch::isnan(ref_features.node_features).sum().item<int>()==0);
            return true;
        }
        else{
            return false;
        }
    }

    bool LoopDetector::encode_concat_sgs(const int& Nr, const DataDict& ref_data_dict,
                                        const int& Ns, const DataDict& src_data_dict,bool fused)
    {
        // Concatenate data
        std::vector<Eigen::Vector3d> xyz;
        std::vector<int> length_vec;
        std::vector<uint32_t> labels;
        std::vector<Eigen::Vector3d> centroids;
        std::vector<uint32_t> nodes;

        int Xr=ref_data_dict.xyz.size(), Xs=src_data_dict.xyz.size();
        // int Nr=ref_nodes.size(), Ns=src_nodes.size();
        if(Xr<1||Xs<1||Nr<1||Ns<1) return false;
        
        xyz.insert(xyz.end(), ref_data_dict.xyz.begin(), ref_data_dict.xyz.end());
        xyz.insert(xyz.end(), src_data_dict.xyz.begin(), src_data_dict.xyz.end());
        length_vec = {Xr, Xs};

        labels.insert(labels.end(), ref_data_dict.labels.begin(), ref_data_dict.labels.end());
        std::vector<uint32_t> aligned_src_labels = src_data_dict.labels;
        for(auto &l: aligned_src_labels) l+=Nr; // incorporate src label offset
        labels.insert(labels.end(), aligned_src_labels.begin(), aligned_src_labels.end());

        centroids.insert(centroids.end(), ref_data_dict.centroids.begin(), ref_data_dict.centroids.end());
        centroids.insert(centroids.end(), src_data_dict.centroids.begin(), src_data_dict.centroids.end());

        nodes.insert(nodes.end(), ref_data_dict.nodes.begin(), ref_data_dict.nodes.end());
        std::vector<uint32_t> aligned_src_nodes = src_data_dict.nodes;
        for(auto &n: aligned_src_nodes) n+=Nr; // incorporate src node offset
        nodes.insert(nodes.end(), aligned_src_nodes.begin(), aligned_src_nodes.end());

        std::cout<<"Concat ref and src scene graphs: "<<Xr<<" + "<<Xs<<" = "<<Xr+Xs<<std::endl;

        // Encode shape features
        torch::Tensor stack_shape_features;
        torch::Tensor stack_node_knn_points;
        torch::Tensor stack_node_knn_features;

        shape_encoder->encode(xyz, length_vec, labels, centroids, nodes,
                            stack_shape_features, stack_node_knn_points, stack_node_knn_features, true);
        
        assert(stack_shape_features.size(0)==Nr+Ns);
        assert(stack_node_knn_points.size(0)==Nr+Ns);

        ref_features.shape_embedded = true;
        ref_features.shape_features = stack_shape_features.index({torch::arange(0,Nr).to(torch::kInt64).to(torch::kCUDA)});
        ref_features.node_knn_points = stack_node_knn_points.index({torch::arange(0,Nr).to(torch::kInt64).to(torch::kCUDA)});
        ref_features.node_knn_features = stack_node_knn_features.index({torch::arange(0,Nr).to(torch::kInt64).to(torch::kCUDA)});

        src_features.shape_embedded = true;
        src_features.shape_features = stack_shape_features.index({torch::arange(Nr,Nr+Ns).to(torch::kInt64).to(torch::kCUDA)});
        src_features.node_knn_points = stack_node_knn_points.index({torch::arange(Nr,Nr+Ns).to(torch::kInt64).to(torch::kCUDA)});
        src_features.node_knn_features = stack_node_knn_features.index({torch::arange(Nr,Nr+Ns).to(torch::kInt64).to(torch::kCUDA)});

        if(fused){
            src_features.node_features = torch::cat({src_features.node_features, src_features.shape_features}, 1);
            ref_features.node_features = torch::cat({ref_features.node_features, ref_features.shape_features}, 1);
        }

        return true;
    }

    int LoopDetector::match_nodes(std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                    std::vector<float> &match_scores,bool fused)
    {   
        bool check_ref_nodes = torch::isnan(ref_features.node_features).sum().item<int>()==0;
        assert (check_ref_nodes);

        sgnet->match_nodes(src_features.node_features, ref_features.node_features, match_pairs, match_scores,fused);
        return match_pairs.size();
    }

    int LoopDetector::match_instance_points(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                            std::vector<Eigen::Vector3d> &corr_src_points,
                                            std::vector<Eigen::Vector3d> &corr_ref_points,
                                            std::vector<float> &corr_scores_vec)
    {
        torch::Tensor corr_points;
        int M = match_pairs.size();
        float match_pairs_array[2][M]; //
        for(int i=0;i<M;i++){
            match_pairs_array[0][i] = int(match_pairs[i].first);  // src_node
            match_pairs_array[1][i] = int(match_pairs[i].second); // ref_node
        }

        torch::Tensor src_corr_nodes = torch::from_blob(match_pairs_array[0], {M}).to(torch::kInt64).to(torch::kCUDA);
        torch::Tensor ref_corr_nodes = torch::from_blob(match_pairs_array[1], {M}).to(torch::kInt64).to(torch::kCUDA);

        torch::Tensor src_guided_knn_points = src_features.node_knn_points.index_select(0, src_corr_nodes);
        torch::Tensor src_guided_knn_feats = src_features.node_knn_features.index_select(0, src_corr_nodes);
        torch::Tensor ref_guided_knn_points = ref_features.node_knn_points.index_select(0, ref_corr_nodes);
        torch::Tensor ref_guided_knn_feats = ref_features.node_knn_features.index_select(0, ref_corr_nodes);
        if(M>30)
            open3d::utility::LogWarning(
                                    "Large number of points candidates. Point match can cost >10GB memory on GPU");
        
        assert(src_guided_knn_feats.size(0)>0 && ref_guided_knn_feats.size(0)>0);
        assert(src_guided_knn_feats.size(1)==512 && ref_guided_knn_feats.size(1)==512);

        int C = sgnet->match_points(src_guided_knn_feats, ref_guided_knn_feats, corr_points, corr_scores_vec);

        TORCH_CHECK(src_guided_knn_feats.device().is_cuda(), "src guided knn feats must be a CUDA tensor");

        // std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points;
        if(C>0){
            extract_corr_points(src_guided_knn_points, ref_guided_knn_points, corr_points, corr_src_points, corr_ref_points);
        }

        return C;
    }

    bool LoopDetector::get_active_node_feats(std::vector<std::vector<float>> &node_feats_vector,
                                            int &N, int &D)const
    {
        if(src_features.node_features.size(0)==0) return false;
        torch::Tensor node_feats = src_features.node_features.clone().to(torch::kCPU);
        N = node_feats.size(0);
        D = node_feats.size(1);
        assert(node_feats.dtype()==torch::kFloat32);
        node_feats_vector.reserve(N);
        // std::cout<<"Get active node features: "<<N<<" nodes, "<<D<<" dims\n";

        for(int i=0;i<N;i++){
            torch::Tensor t = node_feats.index({i});
            std::vector<float> feat_vec(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
            node_feats_vector.emplace_back(feat_vec);
        }
        return true;
    }

}
