#include "SGNet.h"


namespace fmfusion
{

int extract_corr_points(const torch::Tensor &src_guided_knn_points, // (N,K,3)
                        const torch::Tensor &ref_guided_knn_points,
                        const torch::Tensor &corr_points, // (C,3),[node_index, src_index, ref_index]
                        std::vector<Eigen::Vector3d> &corr_src_points,
                        std::vector<Eigen::Vector3d> &corr_ref_points)
{
    int C = corr_points.size(0);
    if(C>0){
        using namespace torch::indexing;
        torch::Tensor corr_match_indices = corr_points.index(
                                        {"...",0}).to(torch::kInt32); // (C,)
        torch::Tensor corr_src_indices = corr_points.index(
                                        {"...",1}).to(torch::kInt32); // (C)
        torch::Tensor corr_ref_indices = corr_points.index(
                                        {"...",2}).to(torch::kInt32); // (C)
        assert(corr_match_indices.max()<src_guided_knn_points.size(0));
        assert(corr_src_indices.max()<src_guided_knn_points.size(1));
        assert(corr_ref_indices.max()<ref_guided_knn_points.size(1));

        torch::Tensor corr_src_points_t = src_guided_knn_points.index(
                                        {corr_match_indices,corr_src_indices}).to(torch::kFloat32); // (C,3)
        torch::Tensor corr_ref_points_t = ref_guided_knn_points.index(
                                        {corr_match_indices,corr_ref_indices}).to(torch::kFloat32); // (C,3)
        corr_src_points_t = corr_src_points_t.to(torch::kCPU);
        corr_ref_points_t = corr_ref_points_t.to(torch::kCPU);

        auto corr_src_points_a = corr_src_points_t.accessor<float,2>();
        auto corr_ref_points_a = corr_ref_points_t.accessor<float,2>();
        
        for (int i=0;i<C;i++){
            corr_src_points.push_back({corr_src_points_a[i][0],
                                        corr_src_points_a[i][1],
                                        corr_src_points_a[i][2]});
            corr_ref_points.push_back({corr_ref_points_a[i][0],
                                        corr_ref_points_a[i][1],
                                        corr_ref_points_a[i][2]});
        }

        // std::cout<<"out shape "<< corr_src_points_t.sizes()<<std::endl;
        // timer.Stop();

        // for (int i=0;i<C;i++){
        //     int match_index = corr_points[i][0].item<int>();
        //     int src_index = corr_points[i][1].item<int>();
        //     int ref_index = corr_points[i][2].item<int>();
        //     corr_src_points.push_back({src_guided_knn_points[match_index][src_index][0].item<float>(),
        //                         src_guided_knn_points[match_index][src_index][1].item<float>(),
        //                         src_guided_knn_points[match_index][src_index][2].item<float>()});
        //     corr_ref_points.push_back({ref_guided_knn_points[match_index][ref_index][0].item<float>(),
        //                         ref_guided_knn_points[match_index][ref_index][1].item<float>(),
        //                         ref_guided_knn_points[match_index][ref_index][2].item<float>()});
        // }
        // std::cout<<"extract corr points takes "<<timer.GetDurationInMillisecond()<<" ms\n";
    }
    return C;
}

int check_nan_features(const torch::Tensor &features) 
{
    auto check_feat = torch::sum(torch::isnan(features),1); // (N,)
    int N = features.size(0);
    int wired_nodes = torch::sum(check_feat>0).item<int>();
    return wired_nodes;
}

SgNet::SgNet(const SgNetConfig &config_, const std::string weight_folder, int cuda_number):config(config_)
{
    std::string sgnet_path = weight_folder + "/sgnet.pt";
    std::string bert_path = weight_folder + "/bert_script.pt";
    std::string vocab_path = weight_folder + "/bert-base-uncased-vocab.txt";
    std::string instance_match_light_path = weight_folder + "/instance_match_light.pt";
    std::string instance_match_fused_path = weight_folder + "/instance_match_fused.pt";
    std::string point_match_path = weight_folder + "/point_match_layer.pt";
    cuda_device_string = "cuda:"+std::to_string(cuda_number);
    std::cout<<"Initializing SGNet on "<<cuda_device_string<<"\n";
    torch::Device device(torch::kCUDA, cuda_number);

    try {
        sgnet_lt = torch::jit::load(sgnet_path);
        sgnet_lt.to(device);
        std::cout << "Load encoder from "<< sgnet_path
                <<" on device "<< cuda_device_string << std::endl;
        sgnet_lt.eval();
        torch::jit::optimize_for_inference(sgnet_lt);
        // sgnet_lt.optimize_for_inference();

        // at::cuda::CUDAStream myStream1 = at::cuda::getStreamFromPool(false, 1);
        // at::cuda::setCurrentCUDAStream(myStream1);
        // auto module_list = sgnet_lt.named_modules();
        // auto param_list = sgnet_lt.named_parameters();
        // std::cout<<sgnet_lt.dump_to_str(true,false,false)<<std::endl;
        // Verify sgnet_lt is on correct device

    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    //
    try
    {
        bert_encoder = torch::jit::load(bert_path);
        bert_encoder.to(device);
        std::cout << "Load bert from "<< bert_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //
    try{
        light_match_layer = torch::jit::load(instance_match_light_path);
        light_match_layer.to(device);
        std::cout << "Load light match layer from "<< instance_match_light_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //
    try{
        fused_match_layer = torch::jit::load(instance_match_fused_path);
        fused_match_layer.to(device);
        std::cout << "Load fused match layer from "<< instance_match_fused_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //
    try{
        point_match_layer = torch::jit::load(point_match_path);
        point_match_layer.to(device);
        std::cout << "Load point match layer from "<< point_match_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    // Load bert bow
    bert_bow_ptr = std::make_shared<BertBow>(weight_folder+"/bert_bow.txt",
                                            weight_folder+"/bert_bow.pt", true);
    enable_bert_bow = bert_bow_ptr->is_loaded();
    std::cout<<"enable bert bow: "<<enable_bert_bow<<"\n";

    // Load tokenizer
    tokenizer.reset(radish::TextTokenizerFactory::Create("radish::BertTokenizer"));
    tokenizer->Init(vocab_path);
    std::cout<<"Tokenizer loaded and initialized\n";

    if(config.warm_up_iter>0) warm_up(config.warm_up_iter, true);
    // torch::cuda::synchronize(0);

}

bool SgNet::load_bert(const std::string weight_folder)
{
    std::string bert_path = weight_folder + "/bert_script.pt";
    o3d_utility::Timer timer;
    timer.Start();

    try
    {
        bert_encoder = torch::jit::load(bert_path);
        bert_encoder.to(cuda_device_string);
        std::cout << "Load bert from "<< bert_path << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
    timer.Stop();
    std::cout<<"Load bert time cost (ms): "<<timer.GetDurationInMillisecond()<<"\n";

    return true;
}

void SgNet::warm_up(int inter, bool verbose)
{
    // Generate fake tensor data
    int N = 120;
    const int semantic_dim = 768;
    triplet_verify_mask = torch::zeros({N,config.triplet_number,3}).to(torch::kInt8).to(cuda_device_string);

    // torch::Tensor semantic_embeddings;
    // torch::Tensor boxes, centroids, anchors, corners, corners_mask;
    
    semantic_embeddings = torch::rand({N, semantic_dim}).to(cuda_device_string);
    boxes = torch::rand({N, 3}).to(cuda_device_string);
    centroids = torch::rand({N, 3}).to(cuda_device_string);
    
    // Anchors should be 0,1,2,...,N-1
    anchors = torch::arange(N).to(torch::kInt32).to(cuda_device_string);
    corners = torch::randint(0, N, {N, config.triplet_number, 2}).to(torch::kInt32).to(cuda_device_string);
    corners_mask = torch::ones({N, config.triplet_number}).to(torch::kInt32).to(cuda_device_string);

    // Run the model
    if(verbose) std::cout<<"Warm up SGNet with "<<inter<<" iterations\n";
    for (int i=0;i<inter;i++){
        auto output = sgnet_lt.forward({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask}).toTuple();
    }

    //
    if(verbose) std::cout<<"Warm up SGNet done\n";
}

bool SgNet::graph_encoder(const std::vector<NodePtr> &nodes, torch::Tensor &node_features)
{
    //
    int N = nodes.size();

    float centroids_arr[N][3];
    float boxes_arr[N][3];
    std::vector<std::string> labels;
    float tokens[N][config.token_padding]={};
    float tokens_attention_mask[N][config.token_padding]={};
    std::vector<int> triplet_anchors; // (N',)
    std::vector<std::vector<Corner>> triplet_corners; // (N', triplet_number, 2)
    float timer_array[5];
    open3d::utility::Timer timer;

    labels.reserve(N);

    // Extract node information
    timer.Start();
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


        // Tokenize semantic label
        if(enable_bert_bow){
            labels.emplace_back(node->semantic);
        }
        else{
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
        }

        // Triplet corners
        if (node->corners.size()>0){
            triplet_anchors.push_back(i);
            std::vector<Corner> corner_vector;
            // for (int k=0;k<triplet_number;k++) corner_vector.push_back({0,1}); // fake corner
            node->sample_corners(config.triplet_number, corner_vector, N); // padding with N
            triplet_corners.push_back(corner_vector);
        }
    }
    timer.Stop();
    timer_array[0] = timer.GetDurationInMillisecond();

    // Pack triplets
    timer.Start();
    int N_valid = triplet_anchors.size(); // Node with at least one valid triplet
    float triplet_anchors_arr[N_valid];
    float triplet_corners_arr[N_valid][config.triplet_number][2];
    float triplet_corners_masks[N_valid][config.triplet_number]={}; // initiialized with 0
    for (int i=0;i<N_valid;i++){
        triplet_anchors_arr[i] = triplet_anchors[i];
        for (int j=0;j<config.triplet_number;j++){
            triplet_corners_arr[i][j][0] = triplet_corners[i][j][0];
            triplet_corners_arr[i][j][1] = triplet_corners[i][j][1];
            if(triplet_corners[i][j][0]<N) triplet_corners_masks[i][j] = 1;
        }
    }

    // Input tensors
    // torch::Tensor semantic_embeddings;
    // torch::Tensor boxes, centroids, anchors, corners, corners_mask;
    boxes = torch::from_blob(boxes_arr, {N, 3}).to(cuda_device_string);
    centroids = torch::from_blob(centroids_arr, {N, 3}).to(cuda_device_string);
    anchors = torch::from_blob(triplet_anchors_arr,{N_valid}).to(torch::kInt32).to(cuda_device_string);
    corners = torch::from_blob(triplet_corners_arr, {N_valid, config.triplet_number, 2}).to(torch::kInt32).to(cuda_device_string);
    corners_mask = torch::from_blob(triplet_corners_masks, {N_valid, config.triplet_number}).to(torch::kInt32).to(cuda_device_string);

    timer.Stop();
    timer_array[1] = timer.GetDurationInMillisecond();
    // std::cout<<"Encoding "<<N<<" node features, with "<<N_valid<<" nodes have valid triplets\n";

    timer.Start();
    if(enable_bert_bow){
        bool bert_bow_ret = bert_bow_ptr->query_semantic_features(labels, semantic_embeddings);
        semantic_embeddings = semantic_embeddings.to(cuda_device_string); 
        // std::cout<<"Queried semantic features "<< bert_bow_ret<<"\n";
    }
    else{
        torch::Tensor input_ids = torch::from_blob(tokens, {N, config.token_padding}).to(torch::kInt32).to(cuda_device_string);
        torch::Tensor attention_mask = torch::from_blob(tokens_attention_mask, {N, config.token_padding}).to(torch::kInt32).to(cuda_device_string);
        torch::Tensor token_type_ids = torch::zeros({N, config.token_padding}).to(torch::kInt32).to(cuda_device_string);

        semantic_embeddings = bert_encoder.forward(
                                {input_ids, attention_mask, token_type_ids}).toTensor();

        int semantic_nan = check_nan_features(semantic_embeddings);
        if(semantic_nan>0)
            open3d::utility::LogWarning("Found {:d} nan semantic embeddings", semantic_nan);                       
    }
    std::cout<<"Bert output correct\n";
    timer.Stop();
    timer_array[2] = timer.GetDurationInMillisecond();


    // Graph encoding
    timer.Start();
    assert(semantic_embeddings.device().str()==cuda_device_string);

    auto output = sgnet_lt.forward({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask}).toTuple();
    node_features = output->elements()[0].toTensor();
    triplet_verify_mask = output->elements()[1].toTensor();
    // std::cout<<"Triplet verify mask shape: "<<triplet_verify_mask.sizes()
    //         <<"types: "<< triplet_verify_mask.dtype()<<"\n";
    assert(triplet_verify_mask.sum().item<int>()<5);
    timer.Stop();
    timer_array[3] = timer.GetDurationInMillisecond();    

    // Check nan
    timer.Start();
    int node_nan = check_nan_features(node_features);
    if(node_nan>0){
        open3d::utility::LogWarning("Found {:d} nan node features", node_nan);
        auto nan_mask = torch::isnan(node_features);
        node_features.index_put_({nan_mask.to(torch::kBool)}, 0);
        open3d::utility::LogWarning("Set nan node features to 0");

    }
    timer.Stop();
    timer_array[4] = timer.GetDurationInMillisecond();

    std::cout<<"graph encode time cost (ms): "
            <<timer_array[0]<<", "
            <<timer_array[1]<<", "
            <<timer_array[2]<<", "
            <<timer_array[3]<<", "
            <<timer_array[4]<<"\n";

    return true;
}

void SgNet::match_nodes(const torch::Tensor &src_node_features, const torch::Tensor &ref_node_features,
                            std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, std::vector<float> &match_scores,bool fused)
{
    // Match layer
    int Ns = src_node_features.size(0);
    int Nr = ref_node_features.size(0);
    int Ds = src_node_features.size(1);
    int Dr = ref_node_features.size(1);
    std::cout<<"Matching "<< Ns << " src nodes and "<< Nr << " ref nodes in fused mode:"<<fused<<"\n";
    assert(Ds==Dr);
    int src_nan_sum = torch::isnan(src_node_features).sum().item<int>();
    int ref_nan_sum = torch::isnan(ref_node_features).sum().item<int>();
    assert(src_nan_sum==0 && ref_nan_sum==0);
    // std::cout<<"ref shape: "<<ref_node_features.sizes()<<std::endl; // (Y,C)
    c10::intrusive_ptr<torch::ivalue::Tuple> match_output;

    if(fused)
        match_output = fused_match_layer.forward({src_node_features, ref_node_features}).toTuple();
    else
        match_output = light_match_layer.forward({src_node_features, ref_node_features}).toTuple();

    torch::Tensor matches = match_output->elements()[0].toTensor(); // (M,2)
    torch::Tensor matches_scores = match_output->elements()[1].toTensor(); // (M,)
    torch::Tensor Kn = match_output->elements()[2].toTensor(); // (X,Y)

    matches = matches.to(torch::kCPU);
    matches_scores = matches_scores.to(torch::kCPU);
    // std::cout<<"matches shape: "<<matches.sizes()<<","<<matches_scores.sizes()<<std::endl;
    // std::cout<<"dtype: "<<matches.dtype()<<","<<matches_scores.dtype()<<std::endl;

    auto matches_a = matches.accessor<long,2>();
    auto matches_scores_a = matches_scores.accessor<float,1>();

    int M = matches.size(0);
    std::cout<<"Find "<<M<<" matched pairs\n";

    for (int i=0;i<M;i++){
        float score = matches_scores_a[i];
        if(score>config.instance_match_threshold){
            auto match_pair = std::make_pair(matches_a[i][0], matches_a[i][1]);
            match_pairs.push_back(match_pair);
            match_scores.push_back(score);
        }
    }

    // Check all elements in Kn is not nan
    // std::cout<<Kn<<std::endl;
    auto check_col = torch::sum(torch::isnan(Kn),0);
    // std::cout<<"Check col: "<<check_col<<std::endl;
    for(int j=0;j<check_col.size(0);j++){
        if(check_col[j].item<int>()>0){
            open3d::utility::LogWarning("Ref node {:d} nan in Kn matrix", j);
        }
    }

    int check = torch::sum(torch::isnan(Kn)).item<int>();
    if(check>0){
        open3d::utility::LogWarning("Found {:d} nan in Kn matrix", check);
    }


}

int SgNet::match_points(const torch::Tensor &src_guided_knn_feats, 
                        const torch::Tensor &ref_guided_knn_feats,
                        const torch::Tensor &src_guided_knn_points,
                        const torch::Tensor &ref_guided_knn_points,
                        std::vector<Eigen::Vector3d> &corr_src_points,
                        std::vector<Eigen::Vector3d> &corr_ref_points,
                        std::vector<int> &corr_match_indices,
                        std::vector<float> &corr_scores_vec)
{
    std::stringstream msg;
    open3d::utility::Timer timer;
    timer.Start();
    auto match_output = point_match_layer.forward({src_guided_knn_feats, ref_guided_knn_feats}).toTuple();
    torch::Tensor corr_points = match_output->elements()[0].toTensor(); // (C,3),[node_index, src_index, ref_index]
    torch::Tensor matching_scores = match_output->elements()[1].toTensor(); // (M,K,K)
    // std::cout<<"matching score: "<<matching_scores.sizes()<<std::endl;
    timer.Stop();
    msg<< "match "<<timer.GetDurationInMillisecond()<<" ms, ";

    int M = src_guided_knn_feats.size(0);
    int C = corr_points.size(0);
    std::cout<<"Find "<<C<<" matched points\n";

    if(C>0){
        // Indexing corr points
        using namespace torch::indexing;
        torch::Tensor corr_match_indices_t = corr_points.index(
                                        {"...",0}).to(torch::kInt32); // (C,)
        torch::Tensor corr_src_indices = corr_points.index(
                                        {"...",1}).to(torch::kInt32); // (C)
        torch::Tensor corr_ref_indices = corr_points.index(
                                        {"...",2}).to(torch::kInt32); // (C)
        assert(corr_match_indices_t.max()<src_guided_knn_points.size(0));
        assert(corr_src_indices.max()<src_guided_knn_points.size(1));
        assert(corr_ref_indices.max()<ref_guided_knn_points.size(1));

        torch::Tensor corr_src_points_t = src_guided_knn_points.index(
                                        {corr_match_indices_t,corr_src_indices}).to(torch::kFloat32); // (C,3)
        torch::Tensor corr_ref_points_t = ref_guided_knn_points.index(
                                        {corr_match_indices_t,corr_ref_indices}).to(torch::kFloat32); // (C,3)

        // toCPU
        torch::Tensor corr_points_cpu = corr_points.clone().to(torch::kCPU);
        matching_scores = matching_scores.to(torch::kCPU);
        corr_src_points_t = corr_src_points_t.to(torch::kCPU);
        corr_ref_points_t = corr_ref_points_t.to(torch::kCPU);

        // Tensor accessor
        auto corr_src_points_a = corr_src_points_t.accessor<float,2>();
        auto corr_ref_points_a = corr_ref_points_t.accessor<float,2>();        
        auto corr_points_a = corr_points_cpu.accessor<long,2>();
        auto matching_scores_a = matching_scores.accessor<float,3>();

        corr_match_indices = std::vector<int>(C,-1);
        corr_scores_vec = std::vector<float>(C,0.0);
        // msg<<"indexing  "<<timer.GetDurationInMillisecond()<<" ms";

        //
        int min_match_index = 100;
        float min_score = 1.0;
        for (int i=0;i<C;i++){
            int match_index = corr_points_a[i][0];
            int src_index = corr_points_a[i][1];
            int ref_index = corr_points_a[i][2];
            corr_match_indices[i] = match_index;
            corr_scores_vec[i] = matching_scores_a[match_index][src_index][ref_index];

            corr_src_points.push_back({corr_src_points_a[i][0],
                                        corr_src_points_a[i][1],
                                        corr_src_points_a[i][2]});
            corr_ref_points.push_back({corr_ref_points_a[i][0],
                                        corr_ref_points_a[i][1],
                                        corr_ref_points_a[i][2]});
            assert(match_index>=0 && match_index<M);
            // if(match_index<min_match_index) min_match_index = match_index;
            // if(corr_scores_vec[i]<min_score) min_score = corr_scores_vec[i];
        }
        // std::cout<<"Min match index: "<<min_match_index
        //         <<", min score: "<<min_score<<"\n";
        // msg<< ", extract "<<timer.GetDurationInMillisecond()<<" ms";
        std::cout<<msg.str()<<"\n";

    }

    return C;

}

bool SgNet::save_hidden_features(const std::string &dir)
{
    if(semantic_embeddings.size(0)==0){
        open3d::utility::LogWarning("No hidden features to save");
        return false;
    }
    else{
        torch::save({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask},
                    dir);
        return true;
    }
}

} // namespace fmfusion

