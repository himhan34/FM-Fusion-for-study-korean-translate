#ifndef LOOP_DETECTOR_H_
#define LOOP_DETECTOR_H_

#include <torch/script.h> 
#include <torch/torch.h>
#include <array>
#include <iostream>
#include <memory>

#include <mapping/Instance.h>
#include <sgloop/Graph.h>
#include <tokenizer/text_tokenizer.h>
#include <Common.h>

namespace fmfusion
{

int extract_corr_points(const torch::Tensor &src_guided_knn_points,
                        const torch::Tensor &ref_guided_knn_points,
                        const torch::Tensor &corr_points, 
                        std::vector<Eigen::Vector3d> &corr_src_points, 
                        std::vector<Eigen::Vector3d> &corr_ref_points);

class SgNet
{

public:
    SgNet(const SgNetConfig &config_, const std::string weight_folder);
    ~SgNet() {};

    /// @brief Modality encoder and graph encoder.
    /// @param nodes 
    /// @param node_features 
    /// @return  
    bool graph_encoder(const std::vector<NodePtr> &nodes, torch::Tensor &node_features);

    void match_nodes(const torch::Tensor &src_node_features, const torch::Tensor &ref_node_features,
        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, std::vector<float> &match_scores, bool fused=false);

    int match_points(const torch::Tensor &src_guided_knn_feats, const torch::Tensor &ref_guided_knn_feats,
        torch::Tensor &corr_points, torch::Tensor &corr_scores);

private:

    std::shared_ptr<radish::TextTokenizer> tokenizer;
    torch::jit::script::Module bert_encoder;
    torch::jit::script::Module sgnet_lt;
    torch::jit::script::Module light_match_layer;
    torch::jit::script::Module fused_match_layer;
    torch::jit::script::Module point_match_layer;
    // int token_padding;
    // int triplet_number;
    SgNetConfig config;

};




}


#endif