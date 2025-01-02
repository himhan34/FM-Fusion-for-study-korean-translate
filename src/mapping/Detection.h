#ifndef FMFUSION_DETECTION_H
#define FMFUSION_DETECTION_H

#include "opencv2/opencv.hpp" // OpenCV 라이브러리 포함
#include "open3d/utility/IJsonConvertible.h" // Open3D JSON 변환 유틸리티 포함

namespace fmfusion
{
    
typedef std::pair<std::string,float> LabelScore; // 라벨과 점수의 쌍을 정의

struct BoundingBox{
    double u0,v0,u1,v1; // 바운딩 박스의 좌상단(u0, v0) 및 우하단(u1, v1) 좌표
};

class Detection
{
public:
    Detection(const int id); // ID를 받아 Detection 객체를 초기화하는 생성자
    Detection(const std::vector<LabelScore> &labels, const BoundingBox &bbox, const cv::Mat &instances_idxs): 
        labels_(labels), bbox_(bbox), instances_idxs_(instances_idxs) {}; // 라벨, 바운딩 박스, 인스턴스 데이터를 이용한 생성자

    std::string extract_label_string() const; // 라벨 문자열을 추출하는 함수
    const cv::Point get_box_center(){return cv::Point((bbox_.u0+bbox_.u1)/2,(bbox_.v0+bbox_.v1)/2);}; // 바운딩 박스 중심 좌표 반환
    const int get_box_area(){return (bbox_.u1-bbox_.u0)*(bbox_.v1-bbox_.v0);} // 바운딩 박스의 면적 계산

public:
    unsigned int id_; // Detection 객체의 ID
    std::vector<LabelScore> labels_; // 라벨과 점수 리스트
    BoundingBox bbox_; // 바운딩 박스 데이터
    cv::Mat instances_idxs_; // 인스턴스 인덱스 매트릭스 [H,W], CV_8UC1 타입
};

typedef std::shared_ptr<Detection> DetectionPtr; // Detection 객체의 공유 포인터 타입 정의

class DetectionFile: public open3d::utility::IJsonConvertible
{
public:
    DetectionFile(int min_mask, int max_box_area): min_mask_(min_mask), max_box_area_(max_box_area) {}; // 최소 마스크 크기와 최대 박스 영역으로 객체 초기화

    bool updateInstanceMap(const std::string &instance_file);    // 인스턴스 맵을 업데이트하는 함수

public:
    bool ConvertToJsonValue(Json::Value &value) const override {return true;}; // JSON 값으로 변환하는 함수 (더미 구현)
    bool ConvertFromJsonValue(const Json::Value &value) override; // JSON 값에서 객체로 변환하는 함수

protected:
    std::vector<std::string> split_str(const std::string s, const std::string delim); // 문자열을 구분자로 나누는 함수
    int min_mask_; // 최소 마스크 크기
    int max_box_area_; // 최대 박스 영역

public:
    std::vector<DetectionPtr> detections; // Detection 객체들의 리스트
    std::vector<std::string> raw_tags; // 원시 태그 리스트
    std::vector<std::string> filter_tags; // 필터링된 태그 리스트
};

}

#endif //FMFUSION_DETECTION_H
