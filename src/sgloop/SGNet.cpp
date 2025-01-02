#include "SGNet.h"


namespace fmfusion
{

// 일치하는 포인트 추출 함수
int extract_corr_points(const torch::Tensor &src_guided_knn_points, // (N,K,3)
                        const torch::Tensor &ref_guided_knn_points,
                        const torch::Tensor &corr_points, // (C,3), [node_index, src_index, ref_index]
                        std::vector<Eigen::Vector3d> &corr_src_points,
                        std::vector<Eigen::Vector3d> &corr_ref_points)
{
    int C = corr_points.size(0);  // 일치하는 포인트의 개수 (C)
    if(C > 0){
        using namespace torch::indexing;
        
        // 일치하는 포인트의 인덱스를 추출
        torch::Tensor corr_match_indices = corr_points.index(
                                        {"...", 0}).to(torch::kInt32); // (C,)
        torch::Tensor corr_src_indices = corr_points.index(
                                        {"...", 1}).to(torch::kInt32); // (C)
        torch::Tensor corr_ref_indices = corr_points.index(
                                        {"...", 2}).to(torch::kInt32); // (C)
        
        // 인덱스가 범위 내에 있는지 확인
        assert(corr_match_indices.max() < src_guided_knn_points.size(0));
        assert(corr_src_indices.max() < src_guided_knn_points.size(1));
        assert(corr_ref_indices.max() < ref_guided_knn_points.size(1));

        // 일치하는 포인트의 좌표를 추출
        torch::Tensor corr_src_points_t = src_guided_knn_points.index(
                                        {corr_match_indices, corr_src_indices}).to(torch::kFloat32); // (C,3)
        torch::Tensor corr_ref_points_t = ref_guided_knn_points.index(
                                        {corr_match_indices, corr_ref_indices}).to(torch::kFloat32); // (C,3)
        
        // CPU로 이동하여 벡터 형태로 변환
        corr_src_points_t = corr_src_points_t.to(torch::kCPU);
        corr_ref_points_t = corr_ref_points_t.to(torch::kCPU);

        auto corr_src_points_a = corr_src_points_t.accessor<float, 2>();
        auto corr_ref_points_a = corr_ref_points_t.accessor<float, 2>();
        
        // 일치하는 포인트의 좌표를 Eigen::Vector3d 형태로 변환하여 벡터에 저장
        for (int i = 0; i < C; i++){
            corr_src_points.push_back({corr_src_points_a[i][0],
                                        corr_src_points_a[i][1],
                                        corr_src_points_a[i][2]});
            corr_ref_points.push_back({corr_ref_points_a[i][0],
                                        corr_ref_points_a[i][1],
                                        corr_ref_points_a[i][2]});
        }
    }
    return C;  // 일치하는 포인트의 개수 반환
}


int check_nan_features(const torch::Tensor &features) 
{
    // 각 특징에 대해 NaN 값을 체크하고, 각 특성의 NaN 개수를 더함
    auto check_feat = torch::sum(torch::isnan(features), 1); // (N,)
    int N = features.size(0);  // 특징의 개수
    int wired_nodes = torch::sum(check_feat > 0).item<int>();  // NaN 값이 있는 특징을 가진 노드 수 계산
    return wired_nodes;  // NaN 특징이 있는 노드 수 반환
}

SgNet::SgNet(const SgNetConfig &config_, const std::string weight_folder, int cuda_number) : config(config_)
{
    // 모델과 관련된 경로 설정
    std::string sgnet_path = weight_folder + "/sgnet.pt";
    std::string bert_path = weight_folder + "/bert_script.pt";
    std::string vocab_path = weight_folder + "/bert-base-uncased-vocab.txt";
    std::string instance_match_light_path = weight_folder + "/instance_match_light.pt";
    std::string instance_match_fused_path = weight_folder + "/instance_match_fused.pt";
    std::string point_match_path = weight_folder + "/point_match_layer.pt";

    // CUDA 장치 설정
    cuda_device_string = "cuda:" + std::to_string(cuda_number);
    std::cout << "Initializing SGNet on " << cuda_device_string << "\n";
    torch::Device device(torch::kCUDA, cuda_number);

    // SGNet 모델 로드
    try {
        sgnet_lt = torch::jit::load(sgnet_path);  // SGNet 모델 로드
        sgnet_lt.to(device);  // CUDA 장치로 이동
        std::cout << "Load encoder from " << sgnet_path
                  << " on device " << cuda_device_string << std::endl;
        sgnet_lt.eval();  // 평가 모드로 설정
        torch::jit::optimize_for_inference(sgnet_lt);  // 추론 최적화
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";  // 모델 로딩 실패 시 오류 메시지 출력
    }

    // BERT 인코더 로드
    try {
        bert_encoder = torch::jit::load(bert_path);  // BERT 모델 로드
        bert_encoder.to(device);  // CUDA 장치로 이동
        std::cout << "Load bert from " << bert_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';  // BERT 모델 로딩 실패 시 오류 메시지 출력
    }

    // 라이트 매칭 레이어 로드
    try {
        light_match_layer = torch::jit::load(instance_match_light_path);  // 라이트 매칭 레이어 로드
        light_match_layer.to(device);  // CUDA 장치로 이동
        std::cout << "Load light match layer from " << instance_match_light_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';  // 라이트 매칭 레이어 로딩 실패 시 오류 메시지 출력
    }

    // 융합 매칭 레이어 로드
    try {
        fused_match_layer = torch::jit::load(instance_match_fused_path);  // 융합 매칭 레이어 로드
        fused_match_layer.to(device);  // CUDA 장치로 이동
        std::cout << "Load fused match layer from " << instance_match_fused_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';  // 융합 매칭 레이어 로딩 실패 시 오류 메시지 출력
    }

    // 포인트 매칭 레이어 로드
    try {
        point_match_layer = torch::jit::load(point_match_path);  // 포인트 매칭 레이어 로드
        point_match_layer.to(device);  // CUDA 장치로 이동
        std::cout << "Load point match layer from " << point_match_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';  // 포인트 매칭 레이어 로딩 실패 시 오류 메시지 출력
    }

    // BertBow 로드
    bert_bow_ptr = std::make_shared<BertBow>(weight_folder + "/bert_bow.txt",
                                             weight_folder + "/bert_bow.pt", true);
    enable_bert_bow = bert_bow_ptr->is_loaded();  // BertBow 로드 여부 확인
    std::cout << "Enable bert bow: " << enable_bert_bow << "\n";

    // 토크나이저 로드 및 초기화
    tokenizer.reset(radish::TextTokenizerFactory::Create("radish::BertTokenizer"));  // BERT 토크나이저 생성
    tokenizer->Init(vocab_path);  // 어휘 파일로 초기화
    std::cout << "Tokenizer loaded and initialized\n";

    // 워밍업 반복 횟수 설정
    if (config.warm_up_iter > 0) warm_up(config.warm_up_iter, true);  // 워밍업 수행
}

bool SgNet::load_bert(const std::string weight_folder)
{
    std::string bert_path = weight_folder + "/bert_script.pt";  // BERT 모델 경로 설정
    o3d_utility::Timer timer;  // 타이머 객체 생성
    timer.Start();  // 타이머 시작

    try
    {
        bert_encoder = torch::jit::load(bert_path);  // BERT 모델 로드
        bert_encoder.to(cuda_device_string);  // CUDA 장치로 모델 이동
        std::cout << "Load bert from " << bert_path << std::endl;  // 모델 로드 완료 메시지 출력
    }
    catch(const std::exception& e)  // 예외 처리
    {
        std::cerr << e.what() << '\n';  // 오류 메시지 출력
        return false;  // 로딩 실패 시 false 반환
    }
    timer.Stop();  // 타이머 종료
    std::cout << "Load bert time cost (ms): " << timer.GetDurationInMillisecond() << "\n";  // 로딩 시간 출력

    return true;  // 모델 로드 성공 시 true 반환
}

void SgNet::warm_up(int inter, bool verbose)
{
    // 더미 텐서 데이터를 생성
    int N = 120;  // 노드 개수
    const int semantic_dim = 768;  // 의미적 임베딩 차원

    // triplet_verify_mask 텐서 생성 (모든 값은 0)
    triplet_verify_mask = torch::zeros({N, config.triplet_number, 3}).to(torch::kInt8).to(cuda_device_string);

    // 의미적 임베딩, 박스, 중심점, 앵커 등을 무작위로 생성
    semantic_embeddings = torch::rand({N, semantic_dim}).to(cuda_device_string);  // 의미적 임베딩
    boxes = torch::rand({N, 3}).to(cuda_device_string);  // 박스 크기
    centroids = torch::rand({N, 3}).to(cuda_device_string);  // 중심점

    // 앵커는 0, 1, 2, ..., N-1로 설정
    anchors = torch::arange(N).to(torch::kInt32).to(cuda_device_string);
    
    // 삼중항 코너 값은 0부터 N-1까지의 랜덤 값으로 생성
    corners = torch::randint(0, N, {N, config.triplet_number, 2}).to(torch::kInt32).to(cuda_device_string);
    corners_mask = torch::ones({N, config.triplet_number}).to(torch::kInt32).to(cuda_device_string);  // 모든 값은 1로 초기화

    // 모델 실행
    if(verbose) std::cout << "Warm up SGNet with " << inter << " iterations\n";  // 워밍업 반복 횟수 출력
    for (int i = 0; i < inter; i++) {
        // SGNet 모델을 실행하여 출력을 받음
        auto output = sgnet_lt.forward({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask}).toTuple();
    }

    // 워밍업 완료 메시지 출력
    if(verbose) std::cout << "Warm up SGNet done\n";
}

bool SgNet::graph_encoder(const std::vector<NodePtr> &nodes, torch::Tensor &node_features)
{
    int N = nodes.size();  // 노드 개수

    // 각 노드의 중심점, 박스 크기, 라벨 등을 저장할 배열 선언
    float centroids_arr[N][3];
    float boxes_arr[N][3];
    std::vector<std::string> labels;  // 노드의 의미적 라벨을 저장할 벡터
    float tokens[N][config.token_padding] = {};  // 토큰화된 라벨을 저장할 배열
    float tokens_attention_mask[N][config.token_padding] = {};  // 토큰의 어텐션 마스크를 저장할 배열
    std::vector<int> triplet_anchors;  // 유효한 삼중항의 인덱스를 저장할 벡터
    std::vector<std::vector<Corner>> triplet_corners;  // 삼중항 코너를 저장할 벡터
    float timer_array[5];  // 타이머를 측정할 배열
    open3d::utility::Timer timer;  // 타이머 객체

    labels.reserve(N);  // 라벨 벡터의 용량을 노드 수만큼 예약

    // 노드 정보를 추출
    timer.Start();
    for (int i = 0; i < N; i++) {
        const NodePtr node = nodes[i];  // 노드 포인터 참조
        Eigen::Vector3d centroid = node->centroid;  // 노드의 중심점
        Eigen::Vector3d extent = node->bbox_shape;  // 노드의 박스 크기

        // 중심점과 박스 크기를 배열에 저장
        centroids_arr[i][0] = centroid[0];
        centroids_arr[i][1] = centroid[1];
        centroids_arr[i][2] = centroid[2];

        boxes_arr[i][0] = extent[0];
        boxes_arr[i][1] = extent[1];
        boxes_arr[i][2] = extent[2];

        // 라벨을 BERT-BOW로 처리
        if (enable_bert_bow) {
            labels.emplace_back(node->semantic);  // BERT-BOW 사용 시 라벨 추가
        } else {
            // BERT를 사용하여 라벨을 토큰화하고, 어텐션 마스크를 설정
            std::vector<int> label_tokens = tokenizer->Encode(node->semantic);  // 라벨을 토큰화
            tokens[i][0] = 101;  // 시작 토큰
            int k = 1;
            for (auto token : label_tokens) {
                tokens[i][k] = token;  // 토큰 저장
                k++;
            }
            tokens[i][k] = 102;  // 종료 토큰

            // 어텐션 마스크 설정
            for (int iter = 0; iter <= k; iter++) {
                tokens_attention_mask[i][iter] = 1;
            }
        }

        // 삼중항 코너 샘플링
        if (node->corners.size() > 0) {
            triplet_anchors.push_back(i);  // 유효한 삼중항 추가
            std::vector<Corner> corner_vector;
            node->sample_corners(config.triplet_number, corner_vector, N);  // 코너 샘플링
            triplet_corners.push_back(corner_vector);  // 코너 정보 저장
        }
    }
    timer.Stop();
    timer_array[0] = timer.GetDurationInMillisecond();  // 타이머 측정 완료

    // 삼중항 처리
    timer.Start();
    int N_valid = triplet_anchors.size();  // 유효한 삼중항 노드 수
    float triplet_anchors_arr[N_valid];
    float triplet_corners_arr[N_valid][config.triplet_number][2];
    float triplet_corners_masks[N_valid][config.triplet_number] = {};  // 마스크 초기화
    for (int i = 0; i < N_valid; i++) {
        triplet_anchors_arr[i] = triplet_anchors[i];
        for (int j = 0; j < config.triplet_number; j++) {
            triplet_corners_arr[i][j][0] = triplet_corners[i][j][0];
            triplet_corners_arr[i][j][1] = triplet_corners[i][j][1];
            if (triplet_corners[i][j][0] < N) triplet_corners_masks[i][j] = 1;  // 유효한 삼중항에 대해서만 마스크 설정
        }
    }

    // 입력 텐서 생성
    boxes = torch::from_blob(boxes_arr, {N, 3}).to(cuda_device_string);  // 박스 텐서 생성
    centroids = torch::from_blob(centroids_arr, {N, 3}).to(cuda_device_string);  // 중심점 텐서 생성
    anchors = torch::from_blob(triplet_anchors_arr, {N_valid}).to(torch::kInt32).to(cuda_device_string);  // 앵커 텐서 생성
    corners = torch::from_blob(triplet_corners_arr, {N_valid, config.triplet_number, 2}).to(torch::kInt32).to(cuda_device_string);  // 코너 텐서 생성
    corners_mask = torch::from_blob(triplet_corners_masks, {N_valid, config.triplet_number}).to(torch::kInt32).to(cuda_device_string);  // 코너 마스크 텐서 생성

    timer.Stop();
    timer_array[1] = timer.GetDurationInMillisecond();  // 타이머 측정 완료

    // BERT-BOW 또는 BERT를 사용하여 의미적 임베딩 생성
    timer.Start();
    if (enable_bert_bow) {
        bool bert_bow_ret = bert_bow_ptr->query_semantic_features(labels, semantic_embeddings);  // BertBow를 사용하여 임베딩 쿼리
        semantic_embeddings = semantic_embeddings.to(cuda_device_string);  // CUDA 장치로 이동
    } else {
        // BERT로 의미적 임베딩 생성
        torch::Tensor input_ids = torch::from_blob(tokens, {N, config.token_padding}).to(torch::kInt32).to(cuda_device_string);  // 입력 ID 텐서
        torch::Tensor attention_mask = torch::from_blob(tokens_attention_mask, {N, config.token_padding}).to(torch::kInt32).to(cuda_device_string);  // 어텐션 마스크 텐서
        torch::Tensor token_type_ids = torch::zeros({N, config.token_padding}).to(torch::kInt32).to(cuda_device_string);  // 토큰 타입 ID 텐서

        semantic_embeddings = bert_encoder.forward({input_ids, attention_mask, token_type_ids}).toTensor();  // BERT 인코더 실행

        int semantic_nan = check_nan_features(semantic_embeddings);  // NaN 체크
        if (semantic_nan > 0)
            open3d::utility::LogWarning("Found {:d} nan semantic embeddings", semantic_nan);  // NaN이 있는 경우 경고
    }
    std::cout << "Bert output correct\n";  // BERT 출력 정상 확인
    timer.Stop();
    timer_array[2] = timer.GetDurationInMillisecond();  // 타이머 측정 완료

    // 그래프 인코딩
    timer.Start();
    assert(semantic_embeddings.device().str() == cuda_device_string);  // 장치 확인

    // SGNet 모델 실행
    auto output = sgnet_lt.forward({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask}).toTuple();
    node_features = output->elements()[0].toTensor();  // 노드 특징 추출
    triplet_verify_mask = output->elements()[1].toTensor();  // 삼중항 검증 마스크 추출
    assert(triplet_verify_mask.sum().item<int>() < 5);  // 삼중항 검증 마스크의 합이 5 미만인지 확인
    timer.Stop();
    timer_array[3] = timer.GetDurationInMillisecond();  // 타이머 측정 완료

    // NaN 체크
    timer.Start();
    int node_nan = check_nan_features(node_features);  // 노드 특징의 NaN 체크
    if (node_nan > 0) {
        open3d::utility::LogWarning("Found {:d} nan node features", node_nan);  // NaN이 있는 경우 경고
        auto nan_mask = torch::isnan(node_features);  // NaN 마스크 생성
        node_features.index_put_({nan_mask.to(torch::kBool)}, 0);  // NaN 값을 0으로 설정
        open3d::utility::LogWarning("Set nan node features to 0");  // NaN을 0으로 설정
    }
    timer.Stop();
    timer_array[4] = timer.GetDurationInMillisecond();  // 타이머 측정 완료

    // 그래프 인코딩 시간 출력
    std::cout << "graph encode time cost (ms): "
              << timer_array[0] << ", "
              << timer_array[1] << ", "
              << timer_array[2] << ", "
              << timer_array[3] << ", "
              << timer_array[4] << "\n";

    return true;  // 성공적으로 그래프 인코딩 완료
}

void SgNet::match_nodes(const torch::Tensor &src_node_features, const torch::Tensor &ref_node_features,
                            std::vector<std::pair<uint32_t, uint32_t>> &match_pairs, std::vector<float> &match_scores, bool fused)
{
    // 매칭 레이어
    int Ns = src_node_features.size(0);  // 원본 노드 수
    int Nr = ref_node_features.size(0);  // 참조 노드 수
    int Ds = src_node_features.size(1);  // 원본 노드 특징 차원
    int Dr = ref_node_features.size(1);  // 참조 노드 특징 차원
    std::cout << "Matching " << Ns << " src nodes and " << Nr << " ref nodes in fused mode:" << fused << "\n";  // 매칭 상태 출력
    assert(Ds == Dr);  // 원본과 참조 노드의 특징 차원 수가 동일한지 확인

    // NaN 값 체크
    int src_nan_sum = torch::isnan(src_node_features).sum().item<int>();  // 원본 노드 특징에서 NaN 값의 개수
    int ref_nan_sum = torch::isnan(ref_node_features).sum().item<int>();  // 참조 노드 특징에서 NaN 값의 개수
    assert(src_nan_sum == 0 && ref_nan_sum == 0);  // NaN 값이 없는지 확인

    c10::intrusive_ptr<torch::ivalue::Tuple> match_output;  // 매칭 결과 저장 변수

    // fused 모드일 경우, 융합된 매칭 레이어 사용
    if (fused)
        match_output = fused_match_layer.forward({src_node_features, ref_node_features}).toTuple();
    else
        match_output = light_match_layer.forward({src_node_features, ref_node_features}).toTuple();  // 일반 매칭 레이어 사용

    // 매칭 결과 및 점수 추출
    torch::Tensor matches = match_output->elements()[0].toTensor();  // 매칭된 쌍 (M, 2)
    torch::Tensor matches_scores = match_output->elements()[1].toTensor();  // 매칭 점수 (M,)
    torch::Tensor Kn = match_output->elements()[2].toTensor();  // Kn 행렬 (X, Y)

    // CPU로 이동하여 매칭 결과 및 점수 처리
    matches = matches.to(torch::kCPU);
    matches_scores = matches_scores.to(torch::kCPU);

    auto matches_a = matches.accessor<long, 2>();  // 매칭된 쌍에 접근하기 위한 accessor
    auto matches_scores_a = matches_scores.accessor<float, 1>();  // 매칭 점수에 접근하기 위한 accessor

    int M = matches.size(0);  // 매칭된 쌍의 개수
    std::cout << "Find " << M << " matched pairs\n";  // 매칭된 쌍의 개수 출력

    // 매칭된 쌍 중 점수가 임계값을 초과한 것만 저장
    for (int i = 0; i < M; i++) {
        float score = matches_scores_a[i];
        if (score > config.instance_match_threshold) {  // 임계값 초과 시
            auto match_pair = std::make_pair(matches_a[i][0], matches_a[i][1]);  // 매칭된 쌍 저장
            match_pairs.push_back(match_pair);
            match_scores.push_back(score);  // 점수 저장
        }
    }

    // Kn 행렬에서 NaN 값이 있는지 확인
    auto check_col = torch::sum(torch::isnan(Kn), 0);  // 각 열에서 NaN 값의 개수 체크
    for (int j = 0; j < check_col.size(0); j++) {
        if (check_col[j].item<int>() > 0) {  // NaN이 있는 열에 대해 경고 출력
            open3d::utility::LogWarning("Ref node {:d} nan in Kn matrix", j);
        }
    }

    int check = torch::sum(torch::isnan(Kn)).item<int>();  // Kn 행렬에서 NaN 값의 총 개수 체크
    if (check > 0) {  // NaN 값이 존재하면 경고 출력
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
    std::stringstream msg;  // 메시지 스트림
    open3d::utility::Timer timer;  // 타이머 객체 생성
    timer.Start();  // 타이머 시작

    // 포인트 매칭 레이어 실행
    auto match_output = point_match_layer.forward({src_guided_knn_feats, ref_guided_knn_feats}).toTuple();
    torch::Tensor corr_points = match_output->elements()[0].toTensor();  // 일치하는 포인트 (C,3), [node_index, src_index, ref_index]
    torch::Tensor matching_scores = match_output->elements()[1].toTensor();  // 매칭 점수 (M,K,K)

    timer.Stop();  // 타이머 종료
    msg << "match " << timer.GetDurationInMillisecond() << " ms, ";

    int M = src_guided_knn_feats.size(0);  // 원본 노드 수
    int C = corr_points.size(0);  // 일치하는 포인트 수
    std::cout << "Find " << C << " matched points\n";  // 매칭된 포인트 수 출력

    if (C > 0) {
        // 일치하는 포인트 인덱싱
        using namespace torch::indexing;
        torch::Tensor corr_match_indices_t = corr_points.index({"...", 0}).to(torch::kInt32);  // (C,)
        torch::Tensor corr_src_indices = corr_points.index({"...", 1}).to(torch::kInt32);  // (C)
        torch::Tensor corr_ref_indices = corr_points.index({"...", 2}).to(torch::kInt32);  // (C)
        
        assert(corr_match_indices_t.max() < src_guided_knn_points.size(0));  // 원본 포인트 크기 확인
        assert(corr_src_indices.max() < src_guided_knn_points.size(1));  // 원본 포인트 크기 확인
        assert(corr_ref_indices.max() < ref_guided_knn_points.size(1));  // 참조 포인트 크기 확인

        // 일치하는 포인트 좌표 추출
        torch::Tensor corr_src_points_t = src_guided_knn_points.index({corr_match_indices_t, corr_src_indices}).to(torch::kFloat32);  // (C,3)
        torch::Tensor corr_ref_points_t = ref_guided_knn_points.index({corr_match_indices_t, corr_ref_indices}).to(torch::kFloat32);  // (C,3)

        // CPU로 이동하여 메모리에서 처리
        torch::Tensor corr_points_cpu = corr_points.clone().to(torch::kCPU);
        matching_scores = matching_scores.to(torch::kCPU);
        corr_src_points_t = corr_src_points_t.to(torch::kCPU);
        corr_ref_points_t = corr_ref_points_t.to(torch::kCPU);

        // 텐서 어세서 생성
        auto corr_src_points_a = corr_src_points_t.accessor<float, 2>();
        auto corr_ref_points_a = corr_ref_points_t.accessor<float, 2>();        
        auto corr_points_a = corr_points_cpu.accessor<long, 2>();
        auto matching_scores_a = matching_scores.accessor<float, 3>();

        corr_match_indices = std::vector<int>(C, -1);  // 매칭된 포인트 인덱스를 저장할 벡터
        corr_scores_vec = std::vector<float>(C, 0.0);  // 매칭된 포인트의 점수를 저장할 벡터

        int min_match_index = 100;
        float min_score = 1.0;
        // 일치하는 포인트와 점수 추출
        for (int i = 0; i < C; i++) {
            int match_index = corr_points_a[i][0];
            int src_index = corr_points_a[i][1];
            int ref_index = corr_points_a[i][2];
            corr_match_indices[i] = match_index;
            corr_scores_vec[i] = matching_scores_a[match_index][src_index][ref_index];  // 점수 저장

            // 일치하는 포인트 좌표 저장
            corr_src_points.push_back({corr_src_points_a[i][0], corr_src_points_a[i][1], corr_src_points_a[i][2]});
            corr_ref_points.push_back({corr_ref_points_a[i][0], corr_ref_points_a[i][1], corr_ref_points_a[i][2]});
            assert(match_index >= 0 && match_index < M);
        }
        std::cout << msg.str() << "\n";  // 메시지 출력
    }

    return C;  // 일치하는 포인트 수 반환
}

bool SgNet::save_hidden_features(const std::string &dir)
{
    if (semantic_embeddings.size(0) == 0) {  // 임베딩이 없다면 경고 메시지 출력
        open3d::utility::LogWarning("No hidden features to save");
        return false;  // 실패 시 false 반환
    } else {
        // 임베딩과 관련된 데이터를 지정된 경로에 저장
        torch::save({semantic_embeddings, boxes, centroids, anchors, corners, corners_mask}, dir);
        return true;  // 성공적으로 저장 시 true 반환
    }
}

} // namespace fmfusion



