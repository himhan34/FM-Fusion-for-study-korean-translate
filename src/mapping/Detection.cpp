#include "json/json.h"  // JSON 데이터를 처리하기 위한 라이브러리 포함
#include "Detection.h"  // Detection 관련 클래스 정의를 포함하는 헤더 파일 포함

namespace fmfusion  // fmfusion 네임스페이스 정의
{

Detection::Detection(const int id): id_(id) 
{
    // Detection 클래스의 생성자에서 객체 ID를 초기화합니다.
}

std::string Detection::extract_label_string() const
{
    std::stringstream ss;  // 문자열을 처리하기 위한 스트림 객체 생성
    for(auto label_score: labels_){  
        // 레이블과 점수 쌍을 순회합니다.
        ss << label_score.first << "(" << std::fixed 
           << std::setprecision(2) << label_score.second << "), ";
        // 레이블 이름과 소수점 둘째 자리까지의 점수를 추가합니다.
    }
    return ss.str();  // 결과 문자열 반환
}

std::vector<std::string> DetectionFile::split_str(
    const std::string s, const std::string delim) 
{
    std::vector<std::string> list;  // 분할된 문자열을 저장할 벡터 생성
    auto start = 0U;  // 시작 인덱스 초기화
    auto end = s.find(delim);  // 구분자 위치를 찾습니다.
    while (true) {  
        // 문자열이 끝날 때까지 반복합니다.
        list.push_back(s.substr(start, end - start));  
        // 구분자로 나뉜 부분 문자열을 벡터에 추가합니다.
        if (end == std::string::npos)  
            break;  // 구분자를 더 이상 찾을 수 없으면 종료
        start = end + delim.length();  // 다음 부분 문자열의 시작 인덱스를 설정합니다.
        end = s.find(delim, start);  // 다음 구분자 위치를 찾습니다.
    }
    return list;  // 분할된 문자열 목록 반환
}

bool DetectionFile::ConvertFromJsonValue(const Json::Value &value)
{
    using namespace open3d;  // Open3D 네임스페이스 사용

    if (!value.isObject()) {  
        // JSON 값이 객체가 아닌 경우
        std::cerr << "DetectionField read JSON failed: unsupported json format." 
                  << std::endl;
        // 에러 메시지를 출력하고 false 반환
        return false;
    }
    const Json::Value object_array = value["mask"];  
    // "mask" 필드에서 객체 배열을 가져옵니다.

    if (!object_array.isArray() || object_array.size() < 2) {  
        // 객체 배열이 아니거나 요소 개수가 2 미만인 경우
        std::cerr << "DetectionField cannot find valid objects." << std::endl;
        return false;  // 변환 실패 반환
    }

    std::stringstream msg;  // 디버깅 메시지를 위한 스트림 생성

    for (int i = 0; i < object_array.size(); ++i) {  
        // 객체 배열의 각 요소를 순회
        const Json::Value object = object_array[i];  // 현재 객체 가져오기
        int detection_id = object["value"].asInt();  
        // 객체의 "value" 필드에서 ID를 가져옵니다.
        if (detection_id == 0) continue;  // ID가 0인 경우 건너뜁니다.
        auto detection = std::make_shared<Detection>(detection_id);  
        // Detection 객체를 생성합니다.

        auto label_score_map = object["labels"];  
        // 레이블과 점수 데이터를 가져옵니다.
        msg << detection_id << ": ";  
        for (auto it = label_score_map.begin(); it != label_score_map.end(); ++it) {  
            // 레이블-점수 쌍을 순회
            std::string label = it.key().asString();  
            float score = (*it).asFloat();  
            detection->labels_.push_back(std::make_pair(label, score));  
            msg << "(" << label << "," << score << ") ";  
        }

        auto box_array = object["box"];  
        // 객체의 바운딩 박스 정보 가져오기
        detection->bbox_.u0 = box_array[0].asDouble();  
        detection->bbox_.v0 = box_array[1].asDouble();  
        detection->bbox_.u1 = box_array[2].asDouble();  
        detection->bbox_.v1 = box_array[3].asDouble();  

        detections.push_back(detection);  
        // 생성된 Detection 객체를 리스트에 추가
        msg << "bbox(" << detection->bbox_.u0 << "," << detection->bbox_.v0 
            << "," << detection->bbox_.u1 << "," << detection->bbox_.v1 << ") ";
        msg << "\n";  
    }

    auto raw_tag_array = split_str(value["raw_tags"].asString(), ".");  
    // "raw_tags" 필드를 분할하여 벡터로 저장
    msg << "raw tags (";
    for (auto tag : raw_tag_array) {  
        // 분할된 태그를 순회
        raw_tags.push_back(tag.substr(tag.find_first_not_of(" ")));  
        msg << "$" << tag.substr(tag.find_first_not_of(" ")) << "$";  
    }
    msg << ")\n";

    auto filter_tag_array = split_str(value["tags"].asString(), ".");  
    // "tags" 필드를 분할하여 벡터로 저장
    msg << "filter tags (";
    for (auto tag : filter_tag_array) {  
        // 분할된 태그를 순회
        filter_tags.push_back(tag.substr(tag.find_first_not_of(" ")));  
        msg << "$" << tag.substr(tag.find_first_not_of(" ")) << "$";  
    }
    msg << ")\n";

    return true;  // 변환 성공 반환
}

bool DetectionFile::updateInstanceMap(const std::string &instance_file)
{
    const int K = detections.size();  // Detection 객체의 총 개수
    cv::Mat detection_map = cv::imread(instance_file, -1);  
    // 인스턴스 맵 파일 읽기
    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(detection_map, &min, &max, &min_loc, &max_loc);  
    // 인스턴스 맵의 최소/최대값 찾기

    if (max != K) {  
        // 인스턴스 수가 일치하지 않는 경우
        std::cerr << "instance map has different number of instances" 
                  << std::endl;
        return false;  // 업데이트 실패 반환
    }
    assert (max == K), "instance map has different number of instances";

    std::vector<int> to_remove_detections;  
    // 제거할 Detection 객체 인덱스 리스트

    for (int idx = 1; idx <= K; ++idx) {  
        // 인스턴스 맵의 각 ID에 대해 처리
        cv::Mat mask = (detection_map == idx);  
        // 현재 ID에 해당하는 마스크 생성
        assert (detections[idx - 1]->id_ == idx), "detection id is not consistent";
        detections[idx - 1]->instances_idxs_ = mask;  
        // 마스크를 Detection 객체에 저장
        if (cv::countNonZero(mask) < min_mask_ || 
            detections[idx - 1]->get_box_area() > max_box_area_) {  
            // 마스크 크기가 너무 작거나 박스 영역이 초과된 경우
            to_remove_detections.push_back(idx - 1);  
        }
    }

    if (to_remove_detections.size() > 0) {  
        // 제거할 Detection 객체가 있는 경우
        std::cerr << "Remove " << to_remove_detections.size() 
                  << " invalid detections" << std::endl;
        for (int i = to_remove_detections.size() - 1; i >= 0; --i) {  
            // 제거할 객체를 리스트에서 삭제
            detections.erase(detections.begin() + to_remove_detections[i]);
        }
    }

    return true;  // 업데이트 성공 반환
}

}
