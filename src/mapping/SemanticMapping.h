#ifndef FMFUSION_SEMANTICMAPPING_H  // 헤더 파일 중복 포함 방지를 위한 매크로 정의 시작
#define FMFUSION_SEMANTICMAPPING_H  // 매크로 정의

#include <unordered_map>  // 효율적인 해시 맵을 제공하는 STL 라이브러리
#include <fstream>  // 파일 입출력을 위한 라이브러리

#include "Common.h"  // 공통 헤더 파일 포함
#include "tools/Color.h"  // 색상 관련 유틸리티 포함
#include "tools/Utility.h"  // 기타 유틸리티 함수 포함
#include "Instance.h"  // Instance 클래스 정의 포함
#include "SemanticDict.h"  // SemanticDict 클래스 정의 포함
#include "BayesianLabel.h"  // BayesianLabel 클래스 정의 포함

namespace fmfusion {  // fmfusion 네임스페이스 정의

    class SemanticMapping {  // SemanticMapping 클래스 정의
    public:
        // 생성자: 매핑 및 인스턴스 설정을 초기화합니다.
        SemanticMapping(const MappingConfig &mapping_cfg, const InstanceConfig &instance_cfg);

        ~SemanticMapping() {};  // 소멸자: 기본 소멸자 사용

    public:
        // 프레임 데이터를 통합하여 맵에 적용합니다.
        void integrate(const int &frame_id,
                       const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose,
                       std::vector<DetectionPtr> &detections);

        // 중첩된 인스턴스를 병합합니다.
        int merge_overlap_instances(std::vector<InstanceId> instance_list = std::vector<InstanceId>());

        // 바닥 인스턴스를 병합합니다.
        int merge_floor(bool verbose=false);

        // 구조적 인스턴스를 병합합니다.
        int merge_overlap_structural_instances(bool merge_all = true);

        // 기타 인스턴스를 병합합니다.
        int merge_other_instances(std::vector<InstancePtr> &instances);
        
        // 포인트 클라우드를 추출합니다.
        void extract_point_cloud(const std::vector<InstanceId> instance_list = std::vector<InstanceId>());

        // 각 인스턴스의 바운딩 박스를 추출 및 업데이트합니다.
        void extract_bounding_boxes();

        // 인스턴스를 업데이트합니다.
        int update_instances(const int &cur_frame_id, const std::vector<InstanceId> &instance_list);

        // 모든 데이터와 사전을 새로고침합니다.
        void refresh_all_semantic_dict();

        // 전역 포인트 클라우드를 내보냅니다.
        std::shared_ptr<open3d::geometry::PointCloud> export_global_pcd(bool filter = false, float vx_size = -1.0);

        // 인스턴스 중심점을 내보냅니다.
        std::vector<Eigen::Vector3d> export_instance_centroids(int earliest_frame_id = -1) const;

        // 인스턴스 주석을 내보냅니다.
        std::vector<std::string> export_instance_annotations(int earliest_frame_id = -1) const;

        // 인스턴스 정보를 질의합니다.
        bool query_instance_info(const std::vector<InstanceId> &names,
                                 std::vector<Eigen::Vector3f> &centroids,
                                 std::vector<std::string> &labels);

        // 유효하지 않은 인스턴스를 제거합니다.
        void remove_invalid_instances();

        // 각 인스턴스에 대한 지오메트리를 가져옵니다.
        std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
        get_geometries(bool point_cloud = true, bool bbox = false);

        // 인스턴스 맵이 비어 있는지 확인합니다.
        bool is_empty() { return instance_map.empty(); }

        // 특정 인스턴스를 반환합니다.
        InstancePtr get_instance(const InstanceId &name) { return instance_map[name]; }

        // 포즈를 적용하여 변환합니다.
        void Transform(const Eigen::Matrix4d &pose);

        // 결과를 저장합니다.
        bool Save(const std::string &path);

        // 결과를 로드합니다.
        bool load(const std::string &path);

        // 인스턴스를 필터링 후 내보냅니다.
        void export_instances(std::vector<InstanceId> &names, std::vector<InstancePtr> &instances,
                              int earliest_frame_id = 0);

    protected:
        // 데이터 연관 작업을 수행합니다.
        int data_association(const std::vector<DetectionPtr> &detections, const std::vector<InstanceId> &active_instances,
                             Eigen::VectorXi &matches,
                             std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs);

        // 새로운 인스턴스를 생성합니다.
        int create_new_instance(const DetectionPtr &detection, const unsigned int &frame_id,
                                const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image,
                                const Eigen::Matrix4d &pose);

        // 활성 인스턴스를 검색합니다.
        std::vector<InstanceId> search_active_instances(const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose,
                                                        const double search_radius = 5.0);

        // 활성 인스턴스를 업데이트합니다.
        void update_active_instances(const std::vector<InstanceId> &active_instances);

        // 최근 인스턴스를 업데이트합니다.
        void update_recent_instances(const int &frame_id,
                                     const std::vector<InstanceId> &active_instances,
                                     const std::vector<InstanceId> &new_instances);

        // 두 레이블 간 유사성을 확인합니다.
        bool IsSemanticSimilar(const std::unordered_map<std::string, float> &measured_labels_a,
                               const std::unordered_map<std::string, float> &measured_labels_b);

        // 두 바운딩 박스 간 2D IoU를 계산합니다.
        double Compute2DIoU(const open3d::geometry::OrientedBoundingBox &box_a,
                            const open3d::geometry::OrientedBoundingBox &box_b);

        // 두 포인트 클라우드 간 3D IoU를 계산합니다.
        double Compute3DIoU(const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation = 1.0);

        // 모호한 인스턴스를 병합합니다.
        int merge_ambiguous_instances(const std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs);

        // 최근 관찰된 인스턴스 목록
        std::unordered_set<InstanceId> recent_instances;

    private:
        // 매핑 및 인스턴스 구성 정보
        MappingConfig mapping_config;
        InstanceConfig instance_config;
        std::unordered_map<InstanceId, InstancePtr> instance_map;
        std::unordered_map<std::string, std::vector<InstanceId>> label_instance_map;
        SemanticDictServer semantic_dict_server;
        BayesianLabel *bayesian_label;

        InstanceId latest_created_instance_id;  // 최근 생성된 인스턴스 ID
        int last_cleanup_frame_id;  // 마지막 클린업 프레임 ID
        int last_update_frame_id;  // 마지막 업데이트 프레임 ID
    };

} // namespace fmfusion

#endif //FMFUSION_SEMANTICMAPPING_H  // 매크로 정의 종료
