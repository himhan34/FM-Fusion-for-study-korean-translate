#ifndef SGNet_H_
#define SGNet_H_

#include <torch/script.h>  // PyTorch 스크립트 API 관련 라이브러리
#include <torch/torch.h>   // PyTorch 텐서 관련 라이브러리
#include <array>           // 배열 관련 라이브러리
#include <iostream>        // 표준 입출력 관련 라이브러리
#include <memory>          // 스마트 포인터 관련 라이브러리

#include <mapping/Instance.h>  // 인스턴스 헤더 파일
#include <sgloop/Graph.h>      // Graph 헤더 파일
#include <sgloop/BertBow.h>    // BertBow 헤더 파일
#include <tokenizer/text_tokenizer.h>  // 텍스트 토크나이저 헤더 파일
#include <Common.h>  // 공통 헤더 파일

namespace fmfusion
{

// 일치하는 포인트 추출 함수
int extract_corr_points(const torch::Tensor &src_guided_knn_points,
                        const torch::Tensor &ref_guided_knn_points,
                        const torch::Tensor &corr_points, 
                        std::vector<Eigen::Vector3d> &corr_src_points, 
                        std::vector<Eigen::Vector3d> &corr_ref_points);

// NaN 특징의 개수를 체크하는 함수
/// @brief  
/// @param features, (N, D) 
/// @return number_nan_features
int check_nan_features(const torch::Tensor &features);

class SgNet
{

public:
    // 생성자: SgNetConfig와 weight 폴더, CUDA 장치 번호를 입력받음
    SgNet(const SgNetConfig &config_, const std::string weight_folder, int cuda_number_=0);
    
    // 소멸자
    ~SgNet() {};

    /// \brief 모달리티 인코더와 그래프 인코더 함수
    /// \param nodes 
    /// \param node_features 
    /// \return  
    bool graph_encoder(const std::vector<NodePtr> &nodes, torch::Tensor &node_features);

    // 노드들 간 매칭 함수
    void match_nodes(const torch::Tensor &src_node_features, const torch::Tensor &ref_node_features,
        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, std::vector<float> &match_scores, bool fused=false);

    /// \brief  일치하는 노드들의 포인트 매칭 함수. 각 노드는 512개의 포인트를 샘플링.
    /// \param src_guided_knn_feats (M,512,256)
    /// \param ref_guided_knn_feats (M,512,256)
    /// \param src_guided_knn_points (M,512,3)
    /// \param ref_guided_knn_points (M,512,3)
    /// \param corr_src_points (C,3)
    /// \param corr_ref_points (C,3)
    /// \param corr_match_indices (C,)
    /// \param corr_scores_vec (C,)
    /// \return C, 매칭된 포인트의 개수
    int match_points(const torch::Tensor &src_guided_knn_feats, 
                    const torch::Tensor &ref_guided_knn_feats,
                    const torch::Tensor &src_guided_knn_points,
                    const torch::Tensor &ref_guided_knn_points,
                    std::vector<Eigen::Vector3d> &corr_src_points,
                    std::vector<Eigen::Vector3d> &corr_ref_points,
                    std::vector<int> &corr_match_indices,
                    std::vector<float> &corr_scores_vec);

    // Bert 모델 로드 함수
    bool load_bert(const std::string weight_folder);

    // 온라인 BERT 여부 반환 함수
    bool is_online_bert()const{return !enable_bert_bow;};

    // 숨겨진 특징을 저장하는 함수
    bool save_hidden_features(const std::string &dir);

private:
    /// \brief  SGNet을 가짜 텐서 데이터로 실행하여 모델을 워밍업하는 함수
    /// \param iter 반복 횟수
    void warm_up(int iter=10,bool verbose=true);

private:
    std::string cuda_device_string;  // CUDA 디바이스 문자열
    std::shared_ptr<radish::TextTokenizer> tokenizer;  // 텍스트 토크나이저
    torch::jit::script::Module bert_encoder;  // BERT 인코더
    torch::jit::script::Module sgnet_lt;  // SGNet 레이어
    torch::jit::script::Module light_match_layer;  // 라이트 매칭 레이어
    torch::jit::script::Module fused_match_layer;  // 융합된 매칭 레이어
    torch::jit::script::Module point_match_layer;  // 포인트 매칭 레이어

    std::shared_ptr<BertBow> bert_bow_ptr;  // BertBow 포인터
    bool enable_bert_bow;  // BertBow 활성화 여부

    SgNetConfig config;  // SGNet 설정

private: // 속도 향상을 위한 숨겨진 상태 변수들
    torch::Tensor semantic_embeddings;  // 의미적 임베딩
    torch::Tensor boxes, centroids, anchors, corners, corners_mask;  // 박스, 중심점, 앵커, 코너, 코너 마스크
    torch::Tensor triplet_verify_mask;  // 삼중항 검증 마스크

};

}

#endif
