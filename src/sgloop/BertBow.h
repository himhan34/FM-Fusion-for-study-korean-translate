/*
 * @Author: Glen LIU 
 * @Date: 2024-06-25 16:27:24 
 * @Last Modified by: Glen LIU
 * @Last Modified time: 2024-06-25 16:36:43
 */
/// \file 이 파일은 미리 생성된 단어와 의미적 특징을 로드합니다.
///     N개의 단어 벡터를 주면 NxD 크기의 의미적 특징을 반환합니다.

#ifndef BERTBOW_H
#define BERTBOW_H

#include <fstream>  // 파일 입출력에 필요한 라이브러리
#include <iostream> // 표준 입출력에 필요한 라이브러리
#include <map>      // 맵 자료구조에 필요한 라이브러리
#include <torch/torch.h>  // PyTorch 텐서 관련 라이브러리

namespace fmfusion
{

// 파일에서 바이트를 읽어오는 함수
std::vector<char> get_the_bytes(const std::string &filename);

// BertBow 클래스 정의
class BertBow
{
    public:
        // 생성자: 단어 파일과 특징 파일을 로드하고 CUDA 장치 여부를 설정
        BertBow(const std::string &words_file, const std::string &word_features_file, bool cuda_device_=false);
        
        // 소멸자
        ~BertBow() {};

        /// 단어 N개를 읽고, NxD 특징을 반환하는 함수
        bool query_semantic_features(const std::vector<std::string> &words, 
                                    torch::Tensor &features);
                                    
        // 로드 성공 여부를 반환하는 함수
        bool is_loaded() const {return load_success;};

        // word2int 맵을 외부로 전달하는 함수
        void get_word2int(std::map<std::string, int>& word2int_map)const{
            word2int_map = word2int;
        }

    private:
        // 단어-인덱스 맵을 로드하는 함수
        bool load_word2int(const std::string &word2int_file);
        
        // 단어 특징을 로드하는 함수
        bool load_word_features(const std::string &word_features_file);

        // 초기화 작업을 위한 워밍업 함수
        void warm_up();

    private:
        bool load_success;  // 로드 성공 여부
        bool cuda_device;   // CUDA 장치 사용 여부
        std::map<std::string, int> word2int;  // 단어와 인덱스를 매핑한 맵
        torch::Tensor word_features;  // 단어 특징을 저장할 텐서
        int N;  // 단어 수

        torch::Tensor indices_tensor;  // 인덱스 텐서
};

} // namespace fmfusion

#endif // BERTBOW_H
