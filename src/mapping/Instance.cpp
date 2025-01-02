#include "Instance.h" // Instance 클래스의 헤더 파일 포함

namespace fmfusion { // fmfusion 네임스페이스 정의

    // Instance 클래스 생성자 정의
    Instance::Instance(const InstanceId id, const unsigned int frame_id, const InstanceConfig &config) :
            id_(id), frame_id_(frame_id), update_frame_id(frame_id), 
            config_(config), bayesian_label(false) 
    {
        // SubVolume 객체 생성 및 초기화
        volume_ = new SubVolume(config_.voxel_length, config_.sdf_trunc,
                                open3d::pipelines::integration::TSDFVolumeColorType::RGB8);

        // 포인트 클라우드 및 관련 객체 초기화
        point_cloud = std::make_shared<open3d::geometry::PointCloud>();
        merged_cloud = std::make_shared<open3d::geometry::PointCloud>();
        min_box = std::make_shared<open3d::geometry::OrientedBoundingBox>();
        predicted_label = std::make_pair("unknown", 0.0); // 초기 라벨 설정

        // 난수를 이용해 색상 초기화
        std::srand(std::time(nullptr));
        color_ = Eigen::Vector3d((double) rand() / RAND_MAX, 
                                (double) rand() / RAND_MAX, 
                                (double) rand() / RAND_MAX);

        // 중심 좌표, 법선, 관측 횟수 초기화
        centroid = Eigen::Vector3d(0.0, 0.0, 0.0);
        normal = Eigen::Vector3d(0.0, 0.0, 0.0);
        observation_count = 1;
    }

    // 베이지안 융합 초기화 함수
    void Instance::init_bayesian_fusion(const std::vector<std::string> &label_set)
    {
        semantic_labels = std::vector<std::string>(label_set); // 라벨 리스트 설정
        probability_vector = Eigen::VectorXf::Zero(label_set.size()); // 확률 벡터 초기화
        bayesian_label = true; // 베이지안 라벨 활성화
    }

    // TSDF 볼륨에 데이터 통합 함수
    void Instance::integrate(const int &frame_id,
                             const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image,
                             const Eigen::Matrix4d &pose) {
        volume_->Integrate(*rgbd_image, config_.intrinsic, pose); // TSDF 볼륨에 RGBD 이미지 통합
        frame_id_ = frame_id; // 최신 프레임 ID 업데이트
    }

    // 포인트 클라우드의 통계적 필터링 함수
    void Instance::filter_pointcloud_statistic() {
        if (point_cloud) { // 포인트 클라우드가 유효한 경우
            size_t old_points_number = point_cloud->points_.size(); // 필터링 전 점 개수
            O3d_Cloud_Ptr output_cloud;
            std::tie(output_cloud, std::ignore) = point_cloud->RemoveStatisticalOutliers(20, 2.0); // 통계적 필터링 수행
            point_cloud = output_cloud; // 필터링 결과 업데이트
        }
    }

    // 클러스터 기반 포인트 클라우드 필터링 함수
    bool Instance::filter_pointcloud_by_cluster() {
        if (point_cloud == nullptr) { // 포인트 클라우드가 없는 경우
            return false;
        } else {
            // DBSCAN 클러스터링 수행
            auto labels = point_cloud->ClusterDBSCAN(config_.cluster_eps, config_.cluster_min_points, true);
            std::unordered_map<int, int> cluster_counts; // 클러스터 ID별 점 개수 저장
            const size_t old_points_number = point_cloud->points_.size(); // 필터링 전 점 개수

            // 클러스터 생성
            for (auto label: labels) {
                if (cluster_counts.find(label) == cluster_counts.end()) {
                    cluster_counts[label] = 1; // 새로운 클러스터 추가
                } else {
                    cluster_counts[label] += 1; // 기존 클러스터 점 개수 증가
                }
            }

            // 유효하지 않은 클러스터 찾기
            std::unordered_set<int> invalid_cluster({-1}); // 클러스터 ID -1은 유효하지 않음

            // 유효한 점만 필터링
            size_t k = 0;
            for (size_t i = 0; i < old_points_number; i++) {
                if (invalid_cluster.find(labels[i]) != invalid_cluster.end()) { // 유효하지 않은 점
                    continue;
                } else { // 유효한 점
                    point_cloud->points_[k] = point_cloud->points_[i];
                    if (point_cloud->HasNormals()) point_cloud->normals_[k] = point_cloud->normals_[i];
                    if (point_cloud->HasCovariances()) point_cloud->covariances_[k] = point_cloud->covariances_[i];
                    if (point_cloud->HasColors()) point_cloud->colors_[k] = point_cloud->colors_[i];
                    k++;
                }
            }
            point_cloud->points_.resize(k); // 필터링 후 점 개수 업데이트
            point_cloud->PaintUniformColor(color_); // 포인트 클라우드에 색상 적용
            o3d_utility::LogInfo("Filter point cloud from {} to {}.", old_points_number, k); // 필터링 결과 로그 출력
            return true;
        }
    }
// 최소 바운딩 박스를 생성하는 함수
    void Instance::CreateMinimalBoundingBox() {
        if (get_cloud_size() < 10) { // 포인트 클라우드 크기가 10 미만이면 반환
            return;
        }

        auto complete_cloud = get_complete_cloud(); // 병합된 포인트 클라우드 가져오기

        using namespace open3d::geometry; // Open3D geometry 네임스페이스 사용
        std::shared_ptr<TriangleMesh> mesh;
        std::tie(mesh, std::ignore) = complete_cloud->ComputeConvexHull(false); // 볼록 껍질 계산
        double min_vol = -1;
        min_box->Clear(); // 기존 바운딩 박스 초기화
        PointCloud hull_pcd;

        for (auto &tri : mesh->triangles_) { // 삼각형 순회
            hull_pcd.points_ = mesh->vertices_; // 볼록 껍질 점 설정
            Eigen::Vector3d a = mesh->vertices_[tri(0)];
            Eigen::Vector3d b = mesh->vertices_[tri(1)];
            Eigen::Vector3d c = mesh->vertices_[tri(2)];

            Eigen::Vector3d u = b - a;
            Eigen::Vector3d v = c - a;
            Eigen::Vector3d w = u.cross(v); // 법선 벡터 계산
            v = w.cross(u); // 수직 벡터 계산
            u = u / u.norm();
            v = v / v.norm();
            w = w / w.norm();

            Eigen::Matrix3d m_rot;
            m_rot << u[0], v[0], 0, u[1], v[1], 0, 0, 0, 1; // 회전 행렬 생성 (롤, 피치 제거)
            hull_pcd.Rotate(m_rot.inverse(), a); // 껍질 회전

            const auto aabox = hull_pcd.GetAxisAlignedBoundingBox(); // 축 정렬 바운딩 박스 계산
            double volume = aabox.Volume();
            if (min_vol == -1. || volume < min_vol) { // 최소 부피 갱신
                min_vol = volume;
                *min_box = aabox.GetOrientedBoundingBox(); // 방향성 바운딩 박스 설정
                min_box->Rotate(m_rot, a); // 원래 방향 복원
            }
        }
        min_box->color_ = color_; // 바운딩 박스 색상 설정

        if (predicted_label.first == "floor") { // 바닥 라벨의 경우 특수 처리
            min_box->center_[2] = min_box->center_[2] - min_box->extent_[2] / 2 + 0.05;
            min_box->extent_[2] = 0.02; // 높이 수정
        }
    }

    // 포인트 클라우드 병합 함수
    void Instance::merge_with(const O3d_Cloud_Ptr &other_cloud,
                              const std::unordered_map<std::string, float> &label_measurements,
                              const int &observations_) {
        *merged_cloud += *other_cloud; // 다른 포인트 클라우드 병합
        merged_cloud->VoxelDownSample(config_.voxel_length); // 다운샘플링
        merged_cloud->PaintUniformColor(color_); // 병합된 클라우드에 색상 적용

        for (const auto label_score : label_measurements) { // 라벨 점수 갱신
            if (measured_labels.find(label_score.first) == measured_labels.end()) {
                measured_labels[label_score.first] = label_score.second;
            } else {
                measured_labels[label_score.first] += label_score.second;
            }

            if (!bayesian_label && measured_labels[label_score.first] > predicted_label.second) {
                predicted_label = std::make_pair(label_score.first, measured_labels[label_score.first]); // 예측 라벨 업데이트
            }
        }
        observation_count += observations_; // 관측 횟수 증가

        CreateMinimalBoundingBox(); // 바운딩 박스 갱신
    }

    // 의미 확률 벡터 업데이트 함수
    bool Instance::update_semantic_probability(const Eigen::VectorXf &probability_vector_) {
        if (probability_vector_.size() != probability_vector.size()) { // 크기 확인
            std::cerr << "Probability vector size mismatch.\n";
            return false;
        } else {
            probability_vector += probability_vector_; // 확률 벡터 갱신
            extract_bayesian_prediciton(); // 베이지안 예측 갱신
            return true;
        }
    }

    // 라벨 업데이트 함수
    void Instance::update_label(const DetectionPtr &detection) {
        for (const auto &label_score : detection->labels_) {
            if (measured_labels.find(label_score.first) == measured_labels.end()) {
                measured_labels[label_score.first] = label_score.second;
            } else {
                measured_labels[label_score.first] += label_score.second;
            }

            if (!bayesian_label && measured_labels[label_score.first] > predicted_label.second) {
                predicted_label = std::make_pair(label_score.first, measured_labels[label_score.first]); // 예측 라벨 업데이트
            }
        }
        observation_count++; // 관측 횟수 증가
    }

    // 베이지안 예측 추출 함수
    void Instance::extract_bayesian_prediciton() {
        if (bayesian_label) {
            int max_idx;
            probability_vector.maxCoeff(&max_idx); // 최대 확률 라벨 찾기
            Eigen::VectorXf probability_normalized = probability_vector.normalized(); // 확률 정규화
            predicted_label = std::make_pair(semantic_labels[max_idx], probability_normalized[max_idx]); // 예측 라벨 설정
        } else {
            std::cerr << "Instance " << id_ << " is not in Bayesian fusion mode.\n";
        }
    }

    // 포인트 클라우드 추출 및 저장 함수
    void Instance::extract_write_point_cloud() {
        double voxel_weight_threshold = config_.min_voxel_weight * observation_count; // 최소 가중치 계산
        point_cloud->Clear(); // 기존 클라우드 초기화
        assert(volume_);
        point_cloud = volume_->ExtractPointCloud(); // 포인트 클라우드 추출
        
        if (point_cloud->HasPoints()) {
            point_cloud->VoxelDownSample(config_.voxel_length); // 다운샘플링
            point_cloud->PaintUniformColor(color_); // 색상 적용
            centroid = point_cloud->GetCenter(); // 중심 좌표 계산
        } else {
            std::cerr << "Instance " << id_ << " has no point cloud.\n";
        }
    }

    // 포인트 클라우드 업데이트 함수
    bool Instance::update_point_cloud(int cur_frame_id, int min_frame_gap) {
        if (cur_frame_id - update_frame_id < min_frame_gap) { // 최소 프레임 간격 확인
            return false;
        } else {
            extract_write_point_cloud(); // 포인트 클라우드 추출
            filter_pointcloud_statistic(); // 통계적 필터링
            CreateMinimalBoundingBox(); // 바운딩 박스 생성
            update_frame_id = cur_frame_id; // 프레임 ID 갱신
            return true;
        }
    }

    // 이전 라벨 로드 함수
    void Instance::load_previous_labels(const std::string &labels_str) {
        std::stringstream ss(labels_str); // 문자열 스트림 생성
        std::string label_score;

        while (std::getline(ss, label_score, ',')) { // 라벨 점수 파싱
            std::stringstream ss2(label_score); // label_name(score)
            std::string label;
            float score;
            std::getline(ss2, label, '('); // 라벨 이름 추출
            ss2 >> score; // 점수 추출
            measured_labels[label] = score; // 라벨 점수 저장
            if (score > predicted_label.second) { // 예측 라벨 갱신
                predicted_label = std::make_pair(label, score);
            }
        }
    }

    // 포인트 클라우드 크기 반환 함수
    size_t Instance::get_cloud_size() const {
        if (point_cloud) {
            size_t cloud_size = point_cloud->points_.size(); // 기본 클라우드 크기
            if (merged_cloud->HasPoints()) cloud_size += merged_cloud->points_.size(); // 병합된 클라우드 크기 추가
            return cloud_size;
        } else {
            return 0;
        }
    }

    // 병합된 포인트 클라우드 반환 함수
    O3d_Cloud_Ptr Instance::get_complete_cloud() const {
        if (merged_cloud->HasPoints()) { // 병합된 클라우드가 있는 경우
            auto complete_cloud = std::make_shared<O3d_Cloud>();
            *complete_cloud += *point_cloud;
            *complete_cloud += *merged_cloud;
            complete_cloud->VoxelDownSample(config_.voxel_length); // 다운샘플링
            return complete_cloud;
        } else {
            return point_cloud; // 병합된 클라우드가 없는 경우 기본 클라우드 반환
        }
    }

} // namespace fmfusion
