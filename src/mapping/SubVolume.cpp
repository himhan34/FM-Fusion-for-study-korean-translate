#include "SubVolume.h" // SubVolume 클래스의 헤더 파일 포함

namespace fmfusion // fmfusion 네임스페이스 정의
{

// SubVolume 클래스 생성자
SubVolume::SubVolume(double voxel_length,
                     double sdf_trunc,
                     TSDFVolumeColorType color_type,
                     int volume_unit_resolution,
                     int depth_sampling_stride)
    : ScalableTSDFVolume(voxel_length, 
                        sdf_trunc, 
                        color_type, 
                        volume_unit_resolution, 
                        depth_sampling_stride) {}

// SubVolume 클래스 소멸자
SubVolume::~SubVolume() {}

// query_observed_points 함수 정의
bool SubVolume::query_observed_points(const PointCloudPtr &cloud_scan, // 입력: 스캔 클라우드
                                 PointCloudPtr &cloud_observed,       // 출력: 관측된 클라우드
                                 const float max_dist)               // 최대 거리 값
{
    // 스캔 클라우드의 모든 점에 대해 반복
    for (size_t i = 0; i < cloud_scan->points_.size(); i++)
    {
        Eigen::Vector3d point = cloud_scan->points_[i]; // 현재 점 가져오기

        // 볼륨 유닛 내 점의 위치 계산
        Eigen::Vector3d p_locate =
            point - Eigen::Vector3d(0.5, 0.5, 0.5) * voxel_length_; // 보정된 위치 계산
        Eigen::Vector3i index0 = LocateVolumeUnit(p_locate); // 점이 속한 볼륨 유닛의 인덱스 계산
        auto unit_itr = volume_units_.find(index0); // 해당 유닛 찾기
        if (unit_itr == volume_units_.end()) { // 유닛이 존재하지 않으면 다음 점으로
            continue;
        }

        const auto &volume0 = *unit_itr->second.volume_; // 유닛의 볼륨 가져오기
        Eigen::Vector3i idx0; // 루트 복셀 인덱스
        Eigen::Vector3d p_grid =
                (p_locate - index0.cast<double>() * volume_unit_length_) /
                voxel_length_;  // 복셀 단위 좌표로 변환
        for (int i = 0; i < 3; i++) { // 복셀 인덱스 범위 보정
            idx0(i) = (int)std::floor(p_grid(i));
            if (idx0(i) < 0) idx0(i) = 0;
            if (idx0(i) >= volume_unit_resolution_)
                idx0(i) = volume_unit_resolution_ - 1;
        }

        // 이웃 복셀 탐색
        for (int i = 0; i < 8; i++) {
            float w0 = 0.0f; // 가중치
            float f0 = 0.0f; // TSDF 값
            Eigen::Vector3i idx1 = idx0 + shift[i]; // 현재 이웃 복셀 인덱스 계산
            if (idx1(0) < volume_unit_resolution_ &&
                idx1(1) < volume_unit_resolution_ &&
                idx1(2) < volume_unit_resolution_) { // 인덱스가 유효한 경우
                w0 = volume0.voxels_[volume0.IndexOf(idx1)].weight_; // 가중치 가져오기
                f0 = volume0.voxels_[volume0.IndexOf(idx1)].tsdf_; // TSDF 값 가져오기
            } 
            else { // 인덱스가 유효하지 않은 경우 다른 유닛 탐색
                Eigen::Vector3i index1 = index0;
                for (int j = 0; j < 3; j++) {
                    if (idx1(j) >= volume_unit_resolution_) {
                        idx1(j) -= volume_unit_resolution_;
                        index1(j) += 1;
                    }
                }
                auto unit_itr1 = volume_units_.find(index1);
                if (unit_itr1 != volume_units_.end()) { // 유닛이 존재하면
                    const auto &volume1 = *unit_itr1->second.volume_;
                    w0 = volume1.voxels_[volume1.IndexOf(idx1)].weight_;
                    f0 = volume1.voxels_[volume1.IndexOf(idx1)].tsdf_;
                }
            }
            if (w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f) { // 유효한 TSDF 값 확인
                cloud_observed->points_.push_back(point); // 관측된 점에 추가
                break;
            }
        }

    }
    return true; // 함수 성공 반환
}

// get_centroid 함수 정의
Eigen::Vector3d SubVolume::get_centroid()
{
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero(); // 초기 중심 좌표 (0, 0, 0)
    int count = 0; // 유닛 개수
    for(const auto &unit : volume_units_) { // 모든 유닛 순회
        if (unit.second.volume_) { // 유닛이 유효한 경우
            centroid += unit.second.volume_->origin_; // 유닛의 중심 좌표 누적
            count++;
        }
    }

    if (count > 0) return centroid / count; // 중심 좌표 계산
    else return Eigen::Vector3d::Zero(); // 유닛이 없으면 (0, 0, 0) 반환
}

} // namespace fmfusion
