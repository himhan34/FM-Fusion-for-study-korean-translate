#include "BertBow.h"


namespace fmfusion
{

    std::vector<char> get_the_bytes(const std::string &filename){
        std::ifstream input(filename, std::ios::binary);
        if(!input.is_open()){
            throw std::runtime_error("Cannot open file " + filename + " for reading.");
            return std::vector<char>();
        }

        std::vector<char> bytes(
                (std::istreambuf_iterator<char>(input)),
                (std::istreambuf_iterator<char>()));
        input.close();
        return bytes;
    }


    BertBow::BertBow(const std::string &words_file, const std::string &word_features_file, bool cuda_device_):
        cuda_device(cuda_device_)
    {
        if(load_word2int(words_file) && load_word_features(word_features_file))
        {
            N = word2int.size();
            assert(N==(word_features.size(0)-1));
            load_success = true;
            if(cuda_device) word_features = word_features.to(torch::kCUDA);
            std::cout<<"Load "<< N <<" words features on CUDA\n";
                        // <<cuda_device<<"\n" ;  //A zero feature is padded in the first row.\n"; 
            warm_up();
                              
        }
        else{
            load_success = false;
            std::cerr<<"Error: Bert Bag of Words Loaded Wrong!\n";
        }
    }

    bool BertBow::query_semantic_features(const std::vector<std::string> &words, 
                                    torch::Tensor &features)
    {
        int Q = words.size();
        if(Q<1) return false;
        float indices[Q];
        // float mask[Q]; // mask the invalid query

        for(int i=0; i<Q; i++){
            if(word2int.find(words[i]) == word2int.end()){
                std::cerr<<"Error: word "<<words[i]<<" not found in the dictionary. Todo\n";
                indices[i] = 0;
            }
            else{
                indices[i] = word2int[words[i]];
            } 
        }
 
        indices_tensor = torch::from_blob(indices, {Q}, torch::kFloat32).toType(torch::kInt64);
        // if(cuda_device) indices_tensor = indices_tensor.to(torch::kCUDA);
        indices_tensor = indices_tensor.to(word_features.device());
        features = torch::index_select(word_features, 0, indices_tensor);

        return true;
    }


    bool BertBow::load_word2int(const std::string &word2int_file)
    {
        // std::cout<<"Load word2int file from "<<word2int_file<<"\n";
        std::ifstream file(word2int_file, std::ifstream::in);
        if (!file.is_open()){
            std::cerr << "Error: cannot open file " << word2int_file << std::endl;
            return false;
        }
        else{
            std::string line;
            while (std::getline(file, line)){
                std::string label;
                int index;
                index = std::stoi(line.substr(0, line.find(".")));
                label = line.substr(line.find(".")+1, line.size()-line.find("."));

                word2int[label] = index;
                // std::cout<<index<<":"<<label<<"\n";
            }
            return true;
        }

    }

    bool BertBow::load_word_features(const std::string &word_features_file)
    {

        std::vector<char> bytes = get_the_bytes(word_features_file);
        if(bytes.empty()){
            return false;
        }
        else{
            // std::cout<<"Load word features from "<<word_features_file<<"\n";
            torch::IValue ivalue = torch::pickle_load(bytes);
            word_features = ivalue.toTensor();
            assert(torch::isnan(word_features).sum().item<int>()==0);

            torch::Tensor zero_features = torch::zeros({1, word_features.size(1)});
            word_features = torch::cat({zero_features, word_features}, 0);
            if(cuda_device) word_features = word_features.to(torch::kCUDA);
            // std::cout<<"feature shape "<<word_features.sizes()<<"\n";
            return true;
        }

    }

    void BertBow::warm_up()
    {
        indices_tensor = torch::randint(0, N, {30}).toType(torch::kInt64);
        indices_tensor = indices_tensor.to(word_features.device());
        torch::Tensor tmp_features = torch::index_select(word_features, 0, indices_tensor);
        std::cout<<"Warm up BertBow with random quries \n";
    }

}
