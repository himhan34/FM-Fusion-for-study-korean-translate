/*
 * @Author: Glen LIU 
 * @Date: 2024-06-25 16:27:24 
 * @Last Modified by: Glen LIU
 * @Last Modified time: 2024-06-25 16:36:43
 */
/// \file This file load a pre-generated words and semantic features.
///     Given a vector of N words, it returns a semantic features in NxD.

#ifndef BERTBOW_H
#define BERTBOW_H

#include <fstream>
#include <iostream>
#include <map>
#include <torch/torch.h>


namespace fmfusion
{

std::vector<char> get_the_bytes(const std::string &filename);

class BertBow
{
    public:
        BertBow(const std::string &words_file, const std::string &word_features_file, bool cuda_device_=false);
        ~BertBow() {};

        /// Read N words, and return NxD features.
        bool query_semantic_features(const std::vector<std::string> &words, 
                                    torch::Tensor &features);
                                    
        bool is_loaded() const {return load_success;};

        void get_word2int(std::map<std::string, int>& word2int_map)const{
            word2int_map = word2int;
        }

    private:
        bool load_word2int(const std::string &word2int_file);
        bool load_word_features(const std::string &word_features_file);

        void warm_up();

    private:
        bool load_success;
        bool cuda_device;
        std::map<std::string, int> word2int;
        torch::Tensor word_features;
        int N;

        torch::Tensor indices_tensor;

};

} // namespace fmfusion


#endif // BERTBOW_H

