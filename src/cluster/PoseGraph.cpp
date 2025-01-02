#include "json/json.h"  // JSON 데이터를 처리하기 위한 라이브러리를 포함합니다.
#include "PoseGraph.h"  // PoseGraph 관련 헤더 파일을 포함합니다.

namespace fmfusion  // fmfusion 네임스페이스 정의
{

bool PoseGraphFile::ConvertFromJsonValue(const Json::Value &value) {
    using namespace open3d;  // open3d 네임스페이스를 사용합니다.
    
    if (!value.isObject()) {  // JSON 값이 객체가 아닌 경우를 확인합니다.
        open3d::utility::LogWarning(
                "PoseGraph read JSON failed: unsupported json "
                "format.");  // JSON 형식이 지원되지 않는 경우 경고 메시지를 출력합니다.
        return false;  // 변환 실패를 반환합니다.
    }
    const Json::Value scans_pose = value;  // JSON 값을 scans_pose로 저장합니다.

    if (scans_pose.size() < 1) {  // JSON 객체에 유효한 데이터가 없는지 확인합니다.
        open3d::utility::LogWarning(
                "PoseGraph cannot find valid objects.");  // 유효한 객체를 찾지 못한 경우 경고 메시지를 출력합니다.
        return false;  // 변환 실패를 반환합니다.
    }

    for (auto it = scans_pose.begin(); it != scans_pose.end(); it++) {  
        // JSON 객체를 순회하며 각 항목을 처리합니다.
        std::string scene_name = it.key().asString();  
        // JSON 객체의 키(장면 이름)를 문자열로 변환합니다.
        Eigen::Matrix4d_u T;  
        // 4x4 행렬 T를 정의합니다.
        for (int i = 0; i < 4; ++i) {  // 행렬의 행을 순회합니다.
            for (int j = 0; j < 4; ++j) {  // 행렬의 열을 순회합니다.
                T(i, j) = (*it)[i * 4 + j].asDouble();  
                // JSON 데이터에서 행렬 요소를 추출하여 T에 저장합니다.
            }
        }
        poses_[scene_name] = T;  
        // 장면 이름과 변환 행렬 T를 poses_ 맵에 저장합니다.
    }

    std::cout << "Read " << poses_.size() << " poses.\n";  
    // 변환된 포즈의 개수를 출력합니다.
    return true;  // 변환 성공을 반환합니다.
}

}
