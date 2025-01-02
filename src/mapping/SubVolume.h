#pragma once // 헤더 파일이 중복 포함되지 않도록 방지

#include <memory> // 스마트 포인터를 위한 라이브러리
#include <string> // 문자열을 위한 라이브러리
#include <vector> // 벡터 컨테이너를 위한 라이브러리
#include <unordered_map> // 해시 맵 컨테이너를 위한 라이브러리
#include <unordered_set> // 해시 집합 컨테이너를 위한 라이브러리

#include "open3d/utility/Logging.h" // Open3D 로그 유틸리티 포함
#include "open3d/geometry/PointCloud.h" // Open3D 포인트 클라우드 클래스 포함
#include "open3d/pipelines/integration/UniformTSDFVolume.h" // Uniform TSDF 볼륨 클래스 포함
#include "open3d/pipelines/integration/ScalableTSDFVolume.h" // Scalable TSDF 볼륨 클래스 포함
#include "open3d/pipelines/integration/MarchingCubesConst.h" // Marching Cubes 상수 포함

namespace fmfusion // fmfusion 네임스페이스 정의
{
    // ScalableTSDFVolume 클래스를 별칭으로 정의
    typedef open3d::pipelines::integration::ScalableTSDFVolume ScalableTSDFVolume;
    // PointCloud의 스마트 포인터 타입 정의
    typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;
    // TSDF 볼륨 색상 타입 정의
    typedef open3d::pipelines::integration::TSDFVolumeColorType TSDFVolumeColorType;

    // SubVolume 클래스 정의 (ScalableTSDFVolume 클래스 상속)
    class SubVolume : public ScalableTSDFVolume {
    public:
        // 생성자 정의
        SubVolume(double voxel_length,
                        double sdf_trunc,
                        TSDFVolumeColorType color_type,
                        int volume_unit_resolution = 16,
                        int depth_sampling_stride = 4);
        // 소멸자 정의
        ~SubVolume() override;

    public:
        // 포인트 클라우드에서 가중치 필터링된 클라우드를 추출 (미사용, 주석 처리)
        // std::shared_ptr<geometry::PointCloud> ExtractWeightedPointCloud(const float min_weight=0.0);

        /// @brief 깊이 이미지에서 스캔 클라우드를 생성하고 볼륨에서 관측된 복셀 점을 쿼리
        bool query_observed_points(const PointCloudPtr &cloud_scan, // 입력: 스캔 클라우드
                                PointCloudPtr &cloud_observed, // 출력: 관측된 클라우드
                                const float max_dist=0.98f); // 최대 거리 값

        // todo: 정확하지 않음
        /// @brief 모든 볼륨 유닛의 중심 좌표를 계산
        /// @return 중심 좌표 반환
        Eigen::Vector3d get_centroid();

    protected:
        // 주어진 점의 볼륨 유닛 위치를 계산
        Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3d &point) {
            return Eigen::Vector3i((int)std::floor(point(0) / volume_unit_length_), // x 좌표 계산
                                (int)std::floor(point(1) / volume_unit_length_), // y 좌표 계산
                                (int)std::floor(point(2) / volume_unit_length_)); // z 좌표 계산
        };

        // 주어진 점에서의 표면 법선 계산
        Eigen::Vector3d GetNormalAt(const Eigen::Vector3d &p);

        // 주어진 점에서의 TSDF 값을 반환
        double GetTSDFAt(const Eigen::Vector3d &p);

    };

}
