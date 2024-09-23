#include "LoopDetector.h"


namespace fmfusion
{
    LoopDetector::LoopDetector(LoopDetectorConfig &lcd_config,
                                ShapeEncoderConfig &shape_encoder_config, 
                                SgNetConfig &sgnet_config, 
                                const std::string weight_folder,
                                int cuda_number,
                                std::vector<std::string> ref_graph_names):weight_folder_dir(weight_folder)
    {
        shape_encoder = std::make_shared<ShapeEncoder>(shape_encoder_config, weight_folder, cuda_number);
        sgnet = std::make_shared<SgNet>(sgnet_config, weight_folder, cuda_number);
        config = lcd_config;
        cuda_device_string = "cuda:"+std::to_string(cuda_number);

        for(const auto &name: ref_graph_names){
            ref_graphs[name] = ImplicitGraph {};
            ref_sg_timestamps[name] = -1.0;
        }

        initialize_graph_features();
    }

    void LoopDetector::initialize_graph_features()
    {
        int max_n = 120;
        int D0 = 128;

        src_features.node_features = torch::zeros({max_n, D0}, torch::kFloat32).to(cuda_device_string);

        // ref_features.node_features = torch::zeros({max_n, D0}, torch::kFloat32).to(cuda_device_string);

        for(auto &kv: ref_graphs){
            kv.second.node_features = torch::zeros({max_n, D0}, torch::kFloat32).to(cuda_device_string);
        }

        std::cout<<"Initialize graph node features\n";
    }

    bool LoopDetector::encode_scene_graph(const std::vector<NodePtr> &nodes, ImplicitGraph &graph_features)
    {
        open3d::utility::Timer timer;
        timer.Start();
        sgnet->graph_encoder(nodes, graph_features.node_features);
        timer.Stop();
        // if(!data_dict.nodes.empty()){ // Encode single graph shapes
        //     shape_encoder->encode(data_dict.xyz,data_dict.length_vec,data_dict.labels,data_dict.centroids,data_dict.nodes,
        //                             graph_features.shape_features, graph_features.node_knn_points, graph_features.node_knn_features);
        //     graph_features.shape_embedded = true;        
        //     if(config.fuse_shape){
        //         graph_features.node_features = torch::cat({graph_features.node_features, graph_features.shape_features}, 1);
        //     }
        //     std::cout<<"Encode single graph shapes\n";
        // }

        return true;
    }

    bool LoopDetector::encode_ref_scene_graph(const std::string &ref_name,
                                            const std::vector<NodePtr> &nodes)
    {
        if(ref_graphs.find(ref_name)==ref_graphs.end()){
            open3d::utility::LogWarning("Reference graph name not found. Skip encoding");
            return false;
        }
        ref_graphs[ref_name].shape_embedded = false;
        return encode_scene_graph(nodes, ref_graphs[ref_name]);

        // ref_features.shape_embedded = false;
        // return encode_scene_graph(nodes, ref_features);
    }

    bool LoopDetector::encode_src_scene_graph(const std::vector<NodePtr> &nodes)
    {
        if(sgnet->is_online_bert()) sgnet->load_bert(weight_folder_dir);
        bool ret = encode_scene_graph(nodes, src_features);
        src_features.shape_embedded = false;
        return ret;
    }

    bool LoopDetector::subscribe_ref_coarse_features(const std::string &ref_name,
                                                    const float &cur_timestamp, 
                                                    const std::vector<std::vector<float>> &coarse_features_vec,
                                                    torch::Tensor coarse_features)
    {
        if(ref_graphs.find(ref_name)==ref_graphs.end()){
            open3d::utility::LogWarning("Reference graph name not found. Skip subscribing");
            return false;
        }

        if(cur_timestamp - ref_sg_timestamps[ref_name] > 0.01){
            int N = coarse_features_vec.size();
            int D = coarse_features_vec[0].size();
            // std::cout<<"Update subscribed node features "<<N <<" x "<<D<<"\n";
            float features_array[N][D];
            for(int i=0;i<N;i++){
                std::copy(coarse_features_vec[i].begin(), coarse_features_vec[i].end(), features_array[i]);
            }
            ref_graphs[ref_name].node_features = torch::from_blob(features_array, {N,D}).to(torch::kFloat32).to(cuda_device_string);
            ref_sg_timestamps[ref_name] = cur_timestamp;

            assert(torch::isnan(ref_graphs[ref_name].node_features).sum().item<int>()==0);
            return true;
        }
        else{
            return false;
        }
    }

    bool LoopDetector::encode_concat_sgs(const std::string &ref_name,
                                        const int& Nr, const DataDict& ref_data_dict,
                                        const int& Ns, const DataDict& src_data_dict,
                                        float &encoding_time,
                                        bool fused,
                                        std::string hidden_feat_dir)
    {
        encoding_time = 0.0;
        if(ref_graphs.find(ref_name)==ref_graphs.end()){
            open3d::utility::LogWarning("Reference graph name not found. Skip encoding");
            return false;
        }

        // Concatenate data
        std::vector<Eigen::Vector3d> xyz;
        std::vector<int> length_vec;
        std::vector<uint32_t> labels; // dense point instances
        std::vector<Eigen::Vector3d> centroids;
        std::vector<uint32_t> nodes;

        o3d_utility::Timer timer;
        std::stringstream msg;

        timer.Start();
        int Xr=ref_data_dict.xyz.size(), Xs=src_data_dict.xyz.size();
        if(Xr<1||Xs<1||Nr<1||Ns<1) return false;
        assert(Ns==src_features.node_features.size(0) && Nr==ref_graphs[ref_name].node_features.size(0));

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
                            stack_shape_features, 
                            stack_node_knn_points, stack_node_knn_features, 
                            encoding_time,
                            hidden_feat_dir);
        
        assert(stack_shape_features.size(0)==Nr+Ns);
        assert(stack_node_knn_points.size(0)==Nr+Ns);

        src_features.shape_embedded = true;
        src_features.shape_features = stack_shape_features.index({torch::arange(Nr,Nr+Ns).to(torch::kInt64).to(cuda_device_string)});
        src_features.node_knn_points = stack_node_knn_points.index({torch::arange(Nr,Nr+Ns).to(torch::kInt64).to(cuda_device_string)});
        src_features.node_knn_features = stack_node_knn_features.index({torch::arange(Nr,Nr+Ns).to(torch::kInt64).to(cuda_device_string)});

        ref_graphs[ref_name].shape_embedded = true;
        ref_graphs[ref_name].shape_features = stack_shape_features.index({torch::arange(0,Nr).to(torch::kInt64).to(cuda_device_string)});
        ref_graphs[ref_name].node_knn_points = stack_node_knn_points.index({torch::arange(0,Nr).to(torch::kInt64).to(cuda_device_string)});
        ref_graphs[ref_name].node_knn_features = stack_node_knn_features.index({torch::arange(0,Nr).to(torch::kInt64).to(cuda_device_string)});

        return true;
    }

    int LoopDetector::match_nodes(const std::string &ref_name,
                                std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                std::vector<float> &match_scores,
                                bool fused, std::string dir)
    {   
        if(ref_graphs.find(ref_name)==ref_graphs.end()){
            open3d::utility::LogWarning("Reference graph name not found. Skip matching");
            return 0;
        }
        bool ref_shape_embedded = ref_graphs[ref_name].shape_embedded;
        torch::Tensor ref_gnn_features = ref_graphs[ref_name].node_features.clone();

        bool check_ref_nodes = torch::isnan(ref_gnn_features).sum().item<int>()==0;
        assert (check_ref_nodes);

        torch::Tensor src_node_features , ref_node_features;
        bool check_fused = false;
        if(src_features.shape_embedded && ref_shape_embedded){
            torch::Tensor ref_shape_features = ref_graphs[ref_name].shape_features.clone();
            src_node_features = torch::cat({src_features.node_features.clone(), src_features.shape_features.clone()}, 1);
            ref_node_features = torch::cat({ref_gnn_features, ref_shape_features}, 1);
            check_fused = true;
        }
        else{
            src_node_features = src_features.node_features;
            ref_node_features = ref_gnn_features; //ref_features.node_features;
        }
        
        sgnet->match_nodes(src_node_features, ref_node_features, match_pairs, match_scores,check_fused);
        
        if(dir!=""){
            std::string output_file_dir;
            if(check_fused) output_file_dir = dir+"_nodes_fused.pt";
            else output_file_dir = dir+"_nodes.pt";
            torch::save({src_node_features, ref_node_features}, output_file_dir);
        }
        
        return match_pairs.size();
    }

    int LoopDetector::match_instance_points(const std::string &ref_name,
                                            const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                            std::vector<Eigen::Vector3d> &corr_src_points,
                                            std::vector<Eigen::Vector3d> &corr_ref_points,
                                            std::vector<int> &corr_match_indices,
                                            std::vector<float> &corr_scores_vec,
                                            std::string dir)
    {

        if(ref_graphs.find(ref_name)==ref_graphs.end()){
            open3d::utility::LogWarning("Reference graph name not found. Skip matching");
            return 0;
        }
        if(!src_features.shape_embedded || !ref_graphs[ref_name].shape_embedded){
            open3d::utility::LogWarning("Shape features or one of them are not embedded. Skip point matching");
            return 0;
        }

        int M = match_pairs.size();
        float match_pairs_array[2][M]; //
        for(int i=0;i<M;i++){
            match_pairs_array[0][i] = int(match_pairs[i].first);  // src_node
            match_pairs_array[1][i] = int(match_pairs[i].second); // ref_node
        }

        torch::Tensor src_corr_nodes = torch::from_blob(match_pairs_array[0], {M}).to(torch::kInt64).to(cuda_device_string);
        torch::Tensor ref_corr_nodes = torch::from_blob(match_pairs_array[1], {M}).to(torch::kInt64).to(cuda_device_string);

        //
        if(!src_features.shape_embedded || !ref_graphs[ref_name].shape_embedded){
            open3d::utility::LogWarning("No node knn points. Skip point matching");
            return 0;
        }

        torch::Tensor src_guided_knn_points = src_features.node_knn_points.index_select(0, src_corr_nodes);
        torch::Tensor src_guided_knn_feats = src_features.node_knn_features.index_select(0, src_corr_nodes);
        torch::Tensor ref_guided_knn_points = ref_graphs[ref_name].node_knn_points.index_select(0, ref_corr_nodes);
        torch::Tensor ref_guided_knn_feats = ref_graphs[ref_name].node_knn_features.index_select(0, ref_corr_nodes);
        if(M>30)
            open3d::utility::LogWarning(
                                    "Large number of points candidates. Point match can cost >10GB memory on GPU");
        
        assert(src_guided_knn_feats.size(0)>0 && ref_guided_knn_feats.size(0)>0);
        assert(src_guided_knn_feats.size(1)==512 && ref_guided_knn_feats.size(1)==512);

        // torch::Tensor corr_points;
        int C = sgnet->match_points(src_guided_knn_feats, 
                                    ref_guided_knn_feats, 
                                    src_guided_knn_points,
                                    ref_guided_knn_points,
                                    corr_src_points,
                                    corr_ref_points,
                                    corr_match_indices,
                                    corr_scores_vec);

        TORCH_CHECK(src_guided_knn_feats.device().is_cuda(), "src guided knn feats must be a CUDA tensor");

        if(dir!=""){
            std::string output_file_dir = dir+"_knn_points.pt";
            torch::save({src_guided_knn_feats, ref_guided_knn_feats},
                            output_file_dir);
        }

        // if(C>0){
        //     extract_corr_points(src_guided_knn_points, ref_guided_knn_points, 
        //                     corr_points, corr_src_points, corr_ref_points);
        //     for (const int&index:corr_match_indices){
        //         assert(index>=0 && index<M);
        //     }
        // }

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

    bool LoopDetector::save_middle_features(const std::string &dir)
    {
        std::cout<<"Saving middle features to "<<dir<<"\n";

        sgnet->save_hidden_features(dir);

        return true;
    }

}
