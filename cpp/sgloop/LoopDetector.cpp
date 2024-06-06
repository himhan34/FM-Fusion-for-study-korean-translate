#include "LoopDetector.h"


namespace fmfusion
{
    LoopDetector::LoopDetector(ShapeEncoderConfig &shape_encoder_config, SgNetConfig &sgnet_config, const std::string weight_folder)
    {
        shape_encoder = std::make_shared<ShapeEncoder>(shape_encoder_config, weight_folder);
        sgnet = std::make_shared<SgNet>(sgnet_config, weight_folder);
    }

    bool LoopDetector::encode_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict, ImplicitGraph &graph_features)
    {
        sgnet->graph_encoder(nodes, graph_features.node_features);
        if(!data_dict.nodes.empty()){
            shape_encoder->encode(data_dict.xyz,data_dict.labels,data_dict.centroids,data_dict.nodes,
                                graph_features.shape_features, graph_features.node_knn_points, graph_features.node_knn_features);
            graph_features.shape_embedded = true;
        }

        return true;
    }

    bool LoopDetector::encode_ref_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict)
    {
        return encode_scene_graph(nodes, data_dict, ref_features);
    }

    bool LoopDetector::encode_src_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict)
    {
        return encode_scene_graph(nodes, data_dict, src_features);
    }

    int LoopDetector::match_nodes(std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                    std::vector<float> &match_scores)
    {
        sgnet->match_nodes(src_features.node_features, ref_features.node_features, match_pairs, match_scores);
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

}
