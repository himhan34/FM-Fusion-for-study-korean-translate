#ifndef FMFUSION_POSEGRAPH_H  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 시작
#define FMFUSION_POSEGRAPH_H  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 끝

#include <unordered_map>  // unordered_map 컨테이너를 사용하기 위해 포함
#include "open3d/Open3D.h"  // Open3D 라이브러리를 사용하기 위해 포함
#include "open3d/utility/IJsonConvertible.h"  // JSON 변환 인터페이스를 사용하기 위해 포함

namespace fmfusion  // fmfusion 네임스페이스 정의
{

class PoseGraphFile: public open3d::utility::IJsonConvertible  // JSON 변환을 지원하는 PoseGraphFile 클래스 정의
{
public:
    PoseGraphFile() {};  // 기본 생성자 정의

public:
    bool ConvertToJsonValue(Json::Value &value) const override {return true;};  
    // JSON 값으로 변환하는 가상 함수, 현재는 항상 true를 반환
    bool ConvertFromJsonValue(const Json::Value &value) override;  
    // JSON 값에서 데이터를 변환하는 가상 함수

public:
    std::unordered_map<std::string, Eigen::Matrix4d_u> poses_;  
    // 포즈 데이터를 저장하기 위한 맵, 키는 문자열(장면 이름), 값은 4x4 변환 행렬
};

}  // fmfusion 네임스페이스 끝
#endif  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 끝
