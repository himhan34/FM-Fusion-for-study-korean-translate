#ifndef LOOP_DETECTOR_H_
#define LOOP_DETECTOR_H_

#include <torch/torch.h>
#include <torch/script.h> 
#include <array>
#include <iostream>
#include <memory>
#include <mapping/Instance.h>

#include <tokenizer/text_tokenizer.h>

namespace fmfusion
{

struct LoopDetectorConfig
{
    std::string encoder_path;
    std::string bert_path;
    std::string vocab_path="/home/cliuci/code_ws/OpensetFusion/cpp/tokenizer/bert-base-uncased-vocab.txt";
    int token_padding=8;
};


class LoopDetector
{

public:
    LoopDetector(const LoopDetectorConfig &config);
    ~LoopDetector() {};

    bool encode(const std::vector<InstanceId> &indices, 
                std::vector<InstancePtr> &instances, torch::Tensor &node_features);

private:

    bool run_tokenize_bert(const std::vector<InstancePtr> &instances);

private:

    std::shared_ptr<radish::TextTokenizer> tokenizer;
    torch::jit::script::Module instance_encoder;
    torch::jit::script::Module bert_encoder;
    int token_padding;

};




}


#endif