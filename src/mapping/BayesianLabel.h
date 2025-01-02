#ifndef BAYESIANLABEL_H  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 시작
#define BAYESIANLABEL_H  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 끝

#include <vector>  // 벡터 자료구조를 사용하기 위해 포함
#include <string>  // 문자열 처리를 위해 포함
#include <iostream>  // 입출력 스트림을 사용하기 위해 포함
#include <eigen3/Eigen/Core>  // Eigen 라이브러리의 Core 기능 포함

#include "Detection.h"  // Detection 관련 헤더 파일 포함
#include "tools/Utility.h"  // 유틸리티 함수가 정의된 헤더 파일 포함

namespace fmfusion  // fmfusion 네임스페이스 정의
{

class BayesianLabel  // BayesianLabel 클래스 정의
{
    public:
        /// \brief likelihood 행렬을 파일에서 로드합니다.
        BayesianLabel(const std::string &likelihood_file, bool verbose=false)
        {
            std::cout << "Semantic fusion load likelihood matrix from "
                      << likelihood_file << std::endl;  // 파일 로드 시작 메시지 출력
            if (load_likelihood_matrix(likelihood_file, verbose)) {
                is_loaded = true;  // 로드 성공 시 플래그 설정
            } else {
                std::cerr << "Failed to load likelihood matrix from "
                          << likelihood_file << std::endl;  // 로드 실패 메시지 출력
            }
        };
        ~BayesianLabel() {};  // 소멸자 정의

        /// \brief 측정값으로부터 확률 벡터를 업데이트합니다.
        bool update_measurements(const std::vector<LabelScore> &measurements,
                                 Eigen::VectorXf &probability_vector)
        {
            if (measurements.empty() || !is_loaded) return false;  // 입력 값이 비어 있거나 로드되지 않은 경우 false 반환

            int rows = measurements.size();  // 측정값의 개수를 행 크기로 설정
            int cols = predict_label_vec.size();  // 예측 레이블 벡터의 크기를 열 크기로 설정
            probability_vector = Eigen::VectorXf::Zero(cols);  // 확률 벡터를 0으로 초기화
            for (int i = 0; i < rows; i++) {
                std::string measure_label = measurements[i].first;  // 측정 레이블 추출
                float measure_score = measurements[i].second;  // 측정 점수 추출
                if (measure_label_map.find(measure_label) != measure_label_map.end()) {
                    probability_vector += measure_score 
                                        * likelihood_matrix.row(measure_label_map[measure_label]).transpose();  
                    // 해당 레이블의 likelihood 행을 점수로 가중치 계산하여 확률 벡터에 추가
                }
            }

            return true;  // 업데이트 성공 반환
        }
        
        /// \brief unordered_map 형식의 측정값을 업데이트합니다.
        bool update_measurements(const std::unordered_map<std::string, float> &measurements,
                                 Eigen::VectorXf &probability_vector)
        {
            if (measurements.empty() || !is_loaded) return false;  // 입력 값이 비어 있거나 로드되지 않은 경우 false 반환
            std::vector<LabelScore> measurements_vec;  // LabelScore 벡터로 변환
            for (const auto &label_score : measurements) {
                measurements_vec.emplace_back(std::make_pair(label_score.first, label_score.second));  // 벡터에 추가
            }

            return update_measurements(measurements_vec, probability_vector);  // 벡터 기반 업데이트 함수 호출
        }

        /// \brief 예측 레이블 벡터를 반환합니다.
        std::vector<std::string> get_label_vec() const {
            return predict_label_vec;
        }

        /// \brief 클래스의 총 개수를 반환합니다.
        int get_num_classes() const {
            return predict_label_vec.size();
        }

    private:

        /// \brief 텍스트 파일에서 likelihood 행렬을 로드합니다.
        bool load_likelihood_matrix(const std::string &likelihood_file, bool verbose)
        {
            std::ifstream file(likelihood_file);  // 파일 스트림 생성
            std::stringstream msg;  // 메시지 저장용 스트림
            if (file.is_open()) {
                std::string line;
                std::getline(file, line);  // 첫 번째 줄 (헤더) 읽기
                std::getline(file, line);  // 측정 레이블 읽기
                measure_label_vec = utility::split_str(line, ",");  // 레이블 벡터로 분리

                std::getline(file, line);  // 예측 레이블 읽기
                predict_label_vec = utility::split_str(line, ",");  // 레이블 벡터로 분리
                
                int rows = measure_label_vec.size();  // 측정 레이블 크기
                int cols = predict_label_vec.size();  // 예측 레이블 크기
                std::cout << "rows: " << rows << " cols: " << cols << std::endl;  // 행렬 크기 출력
                likelihood_matrix = Eigen::MatrixXf::Zero(rows, cols);  // 0으로 초기화된 행렬 생성
                for (int i = 0; i < rows; i++) {
                    std::getline(file, line);  // 각 행 데이터 읽기
                    std::vector<std::string> values = utility::split_str(line, ",");  // 값으로 분리
                    for (int j = 0; j < cols; j++) {
                        likelihood_matrix(i, j) = std::stof(values[j]);  // 문자열 값을 float으로 변환하여 행렬에 저장
                    }
                }
                
                msg << "Measured label-set: ";
                for (int i = 0; i < measure_label_vec.size(); i++) {
                    measure_label_map[measure_label_vec[i]] = i;  // 측정 레이블 맵에 저장
                    msg << measure_label_vec[i] << " ";
                }
                msg << "\nPredict label-set: ";
                for (auto label : predict_label_vec) {
                    msg << label << " ";
                }
                msg << "\nLikelihood matrix: " << likelihood_matrix.rows() << "x"
                    << likelihood_matrix.cols() << std::endl;

                assert(measure_label_vec.size() == likelihood_matrix.rows());  // 행 크기 검증
                assert(predict_label_vec.size() == likelihood_matrix.cols());  // 열 크기 검증
                if (verbose) std::cout << msg.str();  // verbose가 true일 경우 메시지 출력
                return true;  // 로드 성공 반환
            }
            else return false;  // 파일 열기 실패 반환
        }

    private:
        bool is_loaded = false;  // likelihood 행렬이 로드되었는지 여부
        std::vector<std::string> predict_label_vec;  // 예측 레이블 벡터
        std::vector<std::string> measure_label_vec;  // 측정 레이블 벡터
        std::map<std::string, int> measure_label_map;  // 측정 레이블 맵
        Eigen::MatrixXf likelihood_matrix;  // likelihood 행렬
};

}

#endif // BAYESIANLABEL_H  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 끝
