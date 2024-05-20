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

namespace fmfusion
{

struct LoopDetectorConfig
{
    int token_padding=8;
    int triplet_number=20; // number of triplets for each node
};


class LoopDetector
{

public:
    LoopDetector(const LoopDetectorConfig &config, const std::string weight_folder);
    ~LoopDetector() {};

    bool graph_encoder(const std::vector<NodePtr> &nodes, torch::Tensor &node_features);

    void detect_loop(const torch::Tensor &src_node_features, const torch::Tensor &ref_node_features,
        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, std::vector<float> &match_scores);

private:

    std::shared_ptr<radish::TextTokenizer> tokenizer;
    torch::jit::script::Module bert_encoder;
    torch::jit::script::Module sgnet_lt;
    torch::jit::script::Module match_layer;
    torch::jit::script::Module test_fn;
    int token_padding;
    int triplet_number;

};




}


#endif