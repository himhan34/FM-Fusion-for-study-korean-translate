#ifndef SHAPE_ENCODER_H_
#define SHAPE_ENCODER_H_

#include <vector>  // 벡터 컨테이너 라이브러리
#include <Eigen/Core>  // Eigen 라이브러리 (선형 대수학 계산)
#include <torch/torch.h>  // PyTorch 텐서 관련 라이브러리
#include <torch/script.h>  // PyTorch 스크립트 API 관련 라이브러리
#include <open3d/Open3D.h>  // Open3D 라이브러리 (3D 데이터 처리)

#include "Common.h"  // 공통 헤더 파일
#include "thirdparty/extensions/cpu/grid_subsampling.h"  // 서브샘플링을 위한 외부 확장
#include "thirdparty/extensions/cpu/radius_neighbors.h"  // 반경 이웃 탐색을 위한 외부 확장

namespace fmfusion
{
    // 반경 이웃 탐색 함수
    at::Tensor radius_search(at::Tensor q_points, 
                        at::Tensor s_points, 
                        at::Tensor q_lengths, 
                        at::Tensor s_lengths, 
                        float radius, 
                        int neighbor_limit);

    // ShapeEncoder 클래스 정의
    class ShapeEncoder
    {
    /// \brief  하나의 씬 그래프에 대한 shape encoder
    public:
        // 생성자: ShapeEncoderConfig와 weight 폴더, CUDA 장치 번호를 입력받음
        ShapeEncoder(const ShapeEncoderConfig &config_, const std::string weight_folder, int cuda_number=0);
        
        // 소멸자
        ~ShapeEncoder(){};

        /// \brief  씬 그래프의 형상(Shape)을 인코딩하는 함수
        /// \param  xyz_        (X,3), 씬 그래프의 포인트 클라우드
        /// \param  labels_     (X,), 각 포인트의 노드 인덱스
        /// \param  centroids_  (N,3), 각 노드의 중심점
        /// \param  nodes       (N,), 각 노드의 인덱스
        void encode(const std::vector<Eigen::Vector3d> &xyz, const std::vector<int> &length_vec, const std::vector<uint32_t> &labels, 
                    const std::vector<Eigen::Vector3d> &centroids_, const std::vector<uint32_t> &nodes,
                    torch::Tensor &node_shape_feats, 
                    torch::Tensor &node_knn_points,
                    torch::Tensor &node_knn_feats,
                    float &encoding_time,
                    std::string hidden_feat_dir="");

    private:
        // 데이터 전처리 및 스택 모드로 준비하는 함수
        void precompute_data_stack_mode(at::Tensor points, at::Tensor lengths,
                                        std::vector<at::Tensor> &points_list,
                                        std::vector<at::Tensor> &lengths_list,
                                        std::vector<at::Tensor> &neighbors_list,
                                        std::vector<at::Tensor> &subsampling_list,
                                        std::vector<at::Tensor> &upsampling_list);

        // f_points와 레이블을 연관시키는 함수
        void associate_f_points(const std::vector<Eigen::Vector3d> &xyz, const std::vector<uint32_t> &labels, 
                    const at::Tensor &points_f, at::Tensor &labels_f);

        // 노드 f_points 샘플링 함수
        void sample_node_f_points(const at::Tensor &labels_f, const std::vector<uint32_t> &nodes,
                                        at::Tensor &node_f_points, int K=512, int padding_mode=0);
    
    private:
        std::string cuda_device_string;  // CUDA 디바이스 문자열
        ShapeEncoderConfig config;  // ShapeEncoder 설정
        torch::jit::script::Module encoder; // 더 이상 사용되지 않음
        torch::jit::script::Module encoder_v2;  // 새로운 버전의 encoder

    };
    
    typedef std::shared_ptr<ShapeEncoder> ShapeEncoderPtr;  // ShapeEncoder의 스마트 포인터 정의
} // namespace fmfusion

#endif
