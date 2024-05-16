#include "LoopDetector.h"


namespace fmfusion 
{

LoopDetector::LoopDetector(const LoopDetectorConfig &config)
{
    std::cout<<"Initializing loop detector\n";
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        instance_encoder = torch::jit::load(config.encoder_path);
        instance_encoder.to(torch::kCUDA);
        std::cout << "Load encoder from "<< config.encoder_path << std::endl;

    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    
    //
    try
    {
        bert_encoder = torch::jit::load(config.bert_path);
        bert_encoder.to(torch::kCUDA);
        std::cout << "Load bert from "<< config.bert_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    // Load tokenizer
    tokenizer.reset(radish::TextTokenizerFactory::Create("radish::BertTokenizer"));
    tokenizer->Init(config.vocab_path);
    std::cout<<" tokenizer loaded and initialized\n";

    //
    token_padding = config.token_padding;

}


bool LoopDetector::encode(const std::vector<InstanceId> &indices, 
                        std::vector<InstancePtr> &instances, torch::Tensor &node_features)
{
    int N = instances.size();
    
    float centroids[N][3];
    float boxes[N][3];
    std::string labels[N];
    float tokens[N][token_padding]={};
    float tokens_attention_mask[N][token_padding]={};
    
    int i=0;
    for (auto inst:instances)
    {
        Eigen::Vector3d centroid = inst->centroid;
        Eigen::Vector3d extent = inst->min_box->extent_;
        centroids[i][0] = centroid[0];
        centroids[i][1] = centroid[1];
        centroids[i][2] = centroid[2];

        boxes[i][0] = extent[0];
        boxes[i][1] = extent[1];
        boxes[i][2] = extent[2];

        std::string label = inst->get_predicted_class().first;
        labels[i] = label;
        std::vector<int> label_tokens = tokenizer->Encode(label);

        tokens[i][0] = 101;
        int k=1;
        for (auto token: label_tokens){
            tokens[i][k] = token;
            k++;
        }
        tokens[i][k] = 102;

        for (int iter=0; iter<=k; iter++)
            tokens_attention_mask[i][iter] = 1;

        i++;
    }

    // Input tensors
    // torch::Tensor semantics_ = torch::zeros({N, 768}).to(torch::kCUDA);
    torch::Tensor centroids_ = torch::from_blob(centroids, {N, 3}).to(torch::kCUDA);
    torch::Tensor boxes_ = torch::from_blob(boxes, {N, 3}).to(torch::kCUDA);
    torch::Tensor anchors_ = torch::zeros({30, 1}).to(torch::kCUDA);
    torch::Tensor corners_ = torch::zeros({30, 20, 2}).to(torch::kCUDA);
    torch::Tensor corners_mask_ = torch::zeros({30, 20}).to(torch::kCUDA);
    torch::Tensor input_ids = torch::from_blob(tokens, {N, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor attention_mask = torch::from_blob(tokens_attention_mask, {N, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor token_type_ids = torch::zeros({N, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    
    //
    /*
    std::vector<std::string> labels_str = {"floor","wall","bookshelf","television","chair"};
    std::cout<<"Test tokenizer\n";
    i=0;
    for (auto label: labels_str){
        std::vector<int> label_tokens = tokenizer->Encode(label);

        tokens[i][0] = 101;
        int k=1;
        for (auto token: label_tokens){
            tokens[i][k] = token;
            k++;
        }
        tokens[i][k] = 102;
        
        for(int iter=0; iter<=k; iter++)
            tokens_attention_mask[i][iter] = 1;
        i++;
    }
    const int token_number = labels_str.size();
    */

    // fake tokens
    // torch::Tensor input_ids = torch::ones({5, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    // std::cout<<input_ids<<std::endl;

    // Create a vector of inputs.
    std::cout<<"Encoding semantic feats\n";
    torch::Tensor semantic_embeddings = bert_encoder.forward({input_ids, attention_mask, token_type_ids}).toTensor();
    // checksum = semantic_embeddings.sum(1);

    // torch::Tensor features 
    node_features = instance_encoder.forward({semantic_embeddings, centroids_, boxes_, anchors_, corners_, corners_mask_}).toTensor();
    // features.to(torch::kCPU);
    // std::cout<<"output shape: "<<features.sizes()<<std::endl;

    return true;
}



} // namespace fmfusion

