#ifndef FMFUSION_INSTANCE_H
#define FMFUSION_INSTANCE_H

#include <list> // 리스트 컨테이너를 사용하기 위한 헤더 파일
#include <string> // 문자열 처리를 위한 헤더 파일

#include "open3d/Open3D.h" // Open3D 라이브러리 포함
#include "Detection.h" // Detection 관련 클래스 포함
#include "Common.h" // 공통 설정 및 타입 정의 포함
#include "SubVolume.h" // SubVolume 클래스 포함

namespace fmfusion { // fmfusion 네임스페이스 정의

    namespace o3d_utility = open3d::utility; // Open3D 유틸리티를 별칭으로 정의

    // Instance 클래스 정의
    class Instance {

    public:
        // Instance 클래스 생성자: ID, 프레임 ID, 구성 설정으로 초기화
        Instance(const InstanceId id, const unsigned int frame_id, const InstanceConfig &config);

        // Instance 클래스 소멸자
        ~Instance() {};

        // 베이지안 융합 초기화 함수
        void init_bayesian_fusion(const std::vector<std::string> &label_set);

    public:
        // TSDF 볼륨에 데이터를 통합하는 함수
        void integrate(const int &frame_id,
                       const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose);

        // 측정된 라벨 기록 및 예측 라벨 업데이트 함수
        void update_label(const DetectionPtr &detection);

        // 볼륨 유닛의 중심 좌표를 빠르게 업데이트하는 함수
        void fast_update_centroid() { centroid = volume_->get_centroid(); };

        // 포인트 클라우드를 업데이트하는 함수
        bool update_point_cloud(int cur_frame_id, int min_frame_gap = 10);

        // 베이지안 라벨에서 확률 벡터를 업데이트하는 함수
        bool update_semantic_probability(const Eigen::VectorXf &probability_vector_);

        // 포인트 클라우드 병합 함수
        void merge_with(const O3d_Cloud_Ptr &other_cloud,
                        const std::unordered_map<std::string, float> &label_measurements, 
                        const int &observations_);

        // 포인트 클라우드 추출 및 저장 함수
        void extract_write_point_cloud();

        // 통계적 필터링을 통한 포인트 클라우드 정리 함수
        void filter_pointcloud_statistic();

        // 클러스터 기반 포인트 클라우드 필터링 함수
        bool filter_pointcloud_by_cluster();

        // 최소 바운딩 박스 생성 함수
        void CreateMinimalBoundingBox();

        // 이전 측정 라벨 기록 함수
        void load_previous_labels(const std::string &labels_str);

        // 관측 횟수 저장 함수
        void load_obser_count(const int &obs_count){
            observation_count = obs_count;
        }

        // 객체 상태를 파일에 저장하는 함수
        void save(const std::string &path);

        // 파일에서 객체 상태를 로드하는 함수
        void load(const std::string &path);

    public:
        // SubVolume 객체 반환 함수
        SubVolume *get_volume() { return volume_; }

        // 예측된 클래스 반환 함수
        LabelScore get_predicted_class() const { 
            return predicted_label;
        }

        // 측정된 라벨 반환 함수
        std::unordered_map<std::string, float> get_measured_labels() const { 
            return measured_labels; 
        }

        // 포인트 클라우드 크기 반환 함수
        size_t get_cloud_size() const;

        // 전체 병합된 포인트 클라우드 반환 함수
        O3d_Cloud_Ptr get_complete_cloud() const;

        // 구성 설정 반환 함수
        InstanceConfig get_config() const { return config_; }

        // 측정된 라벨 문자열로 반환 함수
        std::string get_measured_labels_string() const {
            std::stringstream label_measurements;
            for (const auto &label_score: measured_labels) {
                label_measurements << label_score.first
                                   << "(" << std::fixed << std::setprecision(2) << label_score.second << "),";
            }
            return label_measurements.str();
        }

        // 관측 횟수 반환 함수
        int get_observation_count() const {
            return observation_count;
        }

        // 인스턴스 ID 반환 함수
        InstanceId get_id() const { return id_; }

        // 인스턴스 ID 변경 함수
        void change_id(InstanceId new_id) { 
            id_ = new_id; 
        }

        // 현재 포인트 클라우드 반환 함수
        O3d_Cloud_Ptr get_point_cloud() const;

        // 베이지안 융합 여부 확인 함수
        bool isBayesianFusion() const { return bayesian_label; }

    private:
        // 최대 확률의 라벨을 추출하여 예측 클래스로 설정
        void extract_bayesian_prediciton();

    public:
        unsigned int frame_id_; // 최신 통합된 프레임 ID
        unsigned int update_frame_id; // 포인트 클라우드 및 바운딩 박스 업데이트 프레임 ID
        Eigen::Vector3d color_; // 인스턴스 색상
        std::shared_ptr<cv::Mat> observed_image_mask; // 이미지 평면에 투영된 볼륨 마스크
        SubVolume *volume_; // SubVolume 객체
        O3d_Cloud_Ptr point_cloud; // 포인트 클라우드 데이터
        Eigen::Vector3d centroid; // 중심 좌표
        Eigen::Vector3d normal; // 표면 법선 벡터
        std::shared_ptr<open3d::geometry::OrientedBoundingBox> min_box; // 최소 바운딩 박스

    private:
        InstanceId id_; // 인스턴스 ID (1 이상)
        InstanceConfig config_; // 인스턴스 구성 설정
        std::unordered_map<std::string, float> measured_labels; // 측정된 라벨
        LabelScore predicted_label; // 예측된 라벨
        int observation_count; // 관측 횟수
        O3d_Cloud_Ptr merged_cloud; // 병합된 포인트 클라우드

        // 베이지안 융합을 위한 설정
        std::vector<std::string> semantic_labels; // 의미 라벨 리스트
        Eigen::VectorXf probability_vector; // 확률 벡터
        bool bayesian_label; // 베이지안 라벨 여부
    };

    typedef std::shared_ptr<Instance> InstancePtr; // Instance 클래스의 스마트 포인터 타입 정의

} // namespace fmfusion

#endif // FMFUSION_INSTANCE_H
