#include "LoopDetector.h"


namespace fmfusion 
{

LoopDetector::LoopDetector(const LoopDetectorConfig &config, const std::string weight_folder)
{
    std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR<<"." 
                                    << TORCH_VERSION_MINOR<<"."
                                    <<TORCH_VERSION_PATCH<< std::endl;
    std::cout<<"Initializing loop detector\n";
    std::string sgnet_path = weight_folder + "/sgnet.pt";
    std::string bert_path = weight_folder + "/bert_script.pt";
    std::string vocab_path = weight_folder + "/bert-base-uncased-vocab.txt";
    std::string instance_match_path = weight_folder + "/instance_match.pt";

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        sgnet_lt = torch::jit::load(sgnet_path);
        sgnet_lt.to(torch::kCUDA);
        std::cout << "Load encoder from "<< sgnet_path << std::endl;

    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
    
    //
    try
    {
        bert_encoder = torch::jit::load(bert_path);
        bert_encoder.to(torch::kCUDA);
        std::cout << "Load bert from "<< bert_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //
    try{
        match_layer = torch::jit::load(instance_match_path);
        match_layer.to(torch::kCUDA);
        std::cout << "Load match layer from "<< instance_match_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    // Load tokenizer
    tokenizer.reset(radish::TextTokenizerFactory::Create("radish::BertTokenizer"));
    tokenizer->Init(vocab_path);
    std::cout<<" tokenizer loaded and initialized\n";

    //
    token_padding = config.token_padding;
    triplet_number = config.triplet_number;
}


bool LoopDetector::graph_encoder(const std::vector<NodePtr> &nodes, torch::Tensor &node_features)
{
    //
    int N = nodes.size();
    
    float centroids_arr[N][3];
    float boxes_arr[N][3];
    std::string labels[N];
    float tokens[N][token_padding]={};
    float tokens_attention_mask[N][token_padding]={};
    std::vector<int> triplet_anchors; // (N',)
    std::vector<std::vector<Corner>> triplet_corners; // (N', triplet_number, 2)
    
    // Extract node information
    for (int i=0;i<N;i++)
    {
        // Geometry
        const NodePtr node = nodes[i];
        Eigen::Vector3d centroid = node->centroid;
        Eigen::Vector3d extent = node->bbox_shape;
        centroids_arr[i][0] = centroid[0];
        centroids_arr[i][1] = centroid[1];
        centroids_arr[i][2] = centroid[2];

        boxes_arr[i][0] = extent[0];
        boxes_arr[i][1] = extent[1];
        boxes_arr[i][2] = extent[2];

        // Semantic label
        labels[i] = node->semantic;
        std::vector<int> label_tokens = tokenizer->Encode(node->semantic);

        tokens[i][0] = 101;
        int k=1;
        for (auto token: label_tokens){
            tokens[i][k] = token;
            k++;
        }
        tokens[i][k] = 102;

        for (int iter=0; iter<=k; iter++)
            tokens_attention_mask[i][iter] = 1;
        
        // Triplet corners
        if (node->corners.size()>0){
            triplet_anchors.push_back(i);
            std::vector<Corner> corner_vector;
            // for (int k=0;k<triplet_number;k++) corner_vector.push_back({0,1}); // fake corner
            node->sample_corners(triplet_number, corner_vector, N); // padding with N
            triplet_corners.push_back(corner_vector);
        }
    }
    
    // Pack triplets
    int N_valid = triplet_anchors.size(); // Node with at least one valid triplet
    float triplet_anchors_arr[N_valid];
    float triplet_corners_arr[N_valid][triplet_number][2];
    float triplet_corners_masks[N_valid][triplet_number]={}; // initiialized with 0
    for (int i=0;i<N_valid;i++){
        triplet_anchors_arr[i] = triplet_anchors[i];
        for (int j=0;j<triplet_number;j++){
            triplet_corners_arr[i][j][0] = triplet_corners[i][j][0];
            triplet_corners_arr[i][j][1] = triplet_corners[i][j][1];
            if(triplet_corners[i][j][0]<N) triplet_corners_masks[i][j] = 1;
        }
    }

    // Input tensors
    torch::Tensor boxes = torch::from_blob(boxes_arr, {N, 3}).to(torch::kCUDA);
    torch::Tensor centroids = torch::from_blob(centroids_arr, {N, 3}).to(torch::kCUDA);
    torch::Tensor input_ids = torch::from_blob(tokens, {N, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor attention_mask = torch::from_blob(tokens_attention_mask, {N, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor token_type_ids = torch::zeros({N, token_padding}).to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor anchors = torch::from_blob(triplet_anchors_arr,{N_valid}).to(torch::kInt32).to(torch::kCUDA); //torch::zeros({30, 1}).to(torch::kCUDA);
    torch::Tensor corners = torch::from_blob(triplet_corners_arr, {N_valid, triplet_number, 2}).to(torch::kInt32).to(torch::kCUDA);
    torch::Tensor corners_mask = torch::from_blob(triplet_corners_masks, {N_valid, triplet_number}).to(torch::kInt32).to(torch::kCUDA);    

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

    // Graph encoding
    std::cout<<"Encoding "<<N<<" node features, with "<<N_valid<<" nodes have valid triplets\n";
    torch::Tensor semantic_embeddings = bert_encoder.forward({input_ids, attention_mask, token_type_ids}).toTensor();

    // std::cout<<"Anchor shape: "<<anchors.sizes()<<std::endl;
    // std::cout<<"Corner shape: "<<corners.sizes()<<std::endl;
    auto output = sgnet_lt.forward({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask}).toTuple();
    node_features = output->elements()[0].toTensor();
    torch::Tensor triplet_verify_mask = output->elements()[1].toTensor();
    assert(triplet_verify_mask.sum().item<int>()<5);
    // std::cout<<triplet_verify_mask.sum().item<int>()<<" triplets are invalid\n";
    std::cout<<"Node features encoded: "<<node_features.sizes()<<std::endl;

    return true;
}

void LoopDetector::detect_loop(const torch::Tensor &src_node_features, const torch::Tensor &ref_node_features,
                            std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, std::vector<float> &match_scores)
{
    // Match layer
    std::cout<<"Matching ... \n";
    std::cout<<"src shape: "<<src_node_features.sizes()<<std::endl; // (X,C)
    std::cout<<"ref shape: "<<ref_node_features.sizes()<<std::endl; // (Y,C)

    auto match_output = match_layer.forward({src_node_features, ref_node_features}).toTuple();
    // torch::Tensor assignment = match_output->elements()[0].toTensor(); // 
    // torch::Tensor assignment_scores = match_output->elements()[1].toTensor(); //

    torch::Tensor matches = match_output->elements()[0].toTensor(); // (M,2)
    torch::Tensor matches_scores = match_output->elements()[1].toTensor(); // (M,)
    torch::Tensor Kn = match_output->elements()[2].toTensor(); // (X,Y)

    int matched_pair = matches.size(0);
    //assignment.sum().item<int>();
    std::cout<<"Find "<<matched_pair<<" matched pairs\n";
    // std::cout<<"matches: "<<matches<<std::endl;
    // std::cout<<"matching scores: "<<matches_scores<<std::endl;

    // std::vector<std::pair<uint32_t,uint32_t>> match_pairs;
    for (int i=0;i<matched_pair;i++){
        match_pairs.push_back({matches[i][0].item<int>(), matches[i][1].item<int>()});
        match_scores.push_back(matches_scores[i].item<float>());
    }


    /*
    torch::Tensor matches = match_output->elements()[0].toTensor();
    torch::Tensor matches_scores = match_output->elements()[1].toTensor();
    torch::Tensor score_mat = match_output->elements()[2].toTensor();

    std::cout<<"Find "<<matches.sizes()<<" instance matches"<<std::endl;
    std::cout<<matches_scores <<std::endl;
    std::cout<<score_mat<<std::endl;
    */
}


} // namespace fmfusion

