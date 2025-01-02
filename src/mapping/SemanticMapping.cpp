#include "SemanticMapping.h"  // SemanticMapping 클래스 정의 포함

namespace fmfusion  // fmfusion 네임스페이스 정의
{

// SemanticMapping 클래스의 생성자
SemanticMapping::SemanticMapping(const MappingConfig &mapping_cfg, const InstanceConfig &instance_cfg)
    : mapping_config(mapping_cfg), instance_config(instance_cfg), semantic_dict_server()  // 멤버 초기화
{
    // SceneGraph 서버 초기화 메시지 출력
    open3d::utility::LogInfo("Initialize SceneGraph server");

    // 최근 생성된 인스턴스 ID 초기화
    latest_created_instance_id = 0;

    // 마지막 클린업 및 업데이트 프레임 ID 초기화
    last_cleanup_frame_id = 0;
    last_update_frame_id = 0;

    // 베이지안 세맨틱 설정 여부에 따라 BayesianLabel 초기화
    if(mapping_config.bayesian_semantic){
        bayesian_label = new BayesianLabel(mapping_config.bayesian_semantic_likelihood, true);  // 베이지안 라벨 초기화
    }
    else bayesian_label = nullptr;  // 베이지안 라벨 비활성화 시 nullptr로 설정
}


std::vector<InstanceId> SemanticMapping::search_active_instances(
    const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose, const double search_radius)
{
    // 활성 인스턴스 목록을 저장할 벡터
    std::vector<InstanceId> active_instances;

    // 최소 유효 단위 설정
    const size_t MIN_UNITS = 1;

    // 인스턴스 맵이 비어 있는 경우 빈 활성 인스턴스를 반환
    if (instance_map.empty()) return active_instances;

    // 깊이 클라우드의 중심 좌표를 계산
    Eigen::Vector3d depth_cloud_center = depth_cloud->GetCenter();

    // 검색 반경 내에 포함되는 인스턴스를 식별
    std::vector<InstanceId> target_instances;
    for (auto &instance_j : instance_map) {
        double dist = (depth_cloud_center - instance_j.second->centroid).norm();  // 중심 간 거리 계산
        if (dist < search_radius) {  // 검색 반경 조건 만족 시 대상에 추가
            target_instances.emplace_back(instance_j.first);
        }
    }

    // 병렬 처리를 통해 활성 인스턴스를 탐색
#pragma omp parallel for default(none) shared(depth_cloud, pose, target_instances, active_instances)
    for (const auto &idx : target_instances) {
        // 인스턴스 맵에서 현재 인스턴스를 가져옴
        InstancePtr instance_j = instance_map[idx];
        auto observed_cloud = std::make_shared<O3d_Cloud>();  // 관찰된 포인트 클라우드를 저장할 객체

        // 볼륨에서 관찰된 포인트를 쿼리
        instance_j->get_volume()->query_observed_points(depth_cloud, observed_cloud);

        // 관찰된 포인트의 개수가 최소 활성 포인트 조건을 만족하는 경우
        if (observed_cloud->points_.size() > mapping_config.min_active_points) {
#pragma omp critical  // 다중 쓰레드 접근 방지
            {
                // 관찰된 포인트 클라우드를 기반으로 이미지 마스크 생성
                instance_j->observed_image_mask = utility::PrjectionCloudToDepth(
                    *observed_cloud, pose.inverse(), instance_config.intrinsic, mapping_config.dilation_size);

                // 활성 인스턴스 목록에 추가
                active_instances.emplace_back(idx);
            }
        }
    }

    // 활성 인스턴스 반환
    return active_instances;
}



void SemanticMapping::update_active_instances(const std::vector<InstanceId> &active_instances)
{
    // 활성 인스턴스의 관찰된 이미지 마스크를 초기화
    for (InstanceId j_ : active_instances) {
        auto instance_j = instance_map[j_];  // 활성 인스턴스를 가져옴
        instance_j->observed_image_mask.reset();  // 관찰된 이미지 마스크 초기화
        // instance_j->fast_update_centroid();  // 중심 업데이트 (주석 처리됨)
    }
}

void SemanticMapping::update_recent_instances(const int &frame_id,
                                              const std::vector<InstanceId> &active_instances,
                                              const std::vector<InstanceId> &new_instances)
{
    std::vector<InstanceId> invalid_instances;  // 유효하지 않은 인스턴스 목록 (삭제 예정)

    // 활성 및 신규 인스턴스를 최근 인스턴스 목록에 추가
    for (InstanceId j_ : active_instances) recent_instances.emplace(j_);
    for (InstanceId j_ : new_instances) recent_instances.emplace(j_);

    // 최근 인스턴스 검사 및 오래된 인스턴스 제거
    for (auto idx : recent_instances) {
        auto inst = instance_map.find(idx);  // 인스턴스 맵에서 인스턴스 검색
        if (inst == instance_map.end()) continue;  // 인스턴스가 존재하지 않으면 무시

        // 최근 윈도우 크기 초과 확인
        if ((frame_id - inst->second->frame_id_) > mapping_config.recent_window_size) {
            // 재관측되지 않고 포인트 클라우드가 없는 인스턴스를 제거
            if (!inst->second->point_cloud->HasPoints()) {
                instance_map.erase(inst);  // 인스턴스 맵에서 삭제
            }
        }
    }

    // 무효화된 인스턴스를 최근 인스턴스 목록에서 제거
    for (auto idx : invalid_instances) {
        recent_instances.erase(idx);
    }
}

int SemanticMapping::data_association(const std::vector<DetectionPtr> &detections, 
                                    const std::vector<InstanceId> &active_instances,
                                    Eigen::VectorXi &matches,
                                    std::vector<std::pair<InstanceId,InstanceId>> &ambiguous_pairs)
{
    int K = detections.size();  // 감지된 객체 수
    int M = active_instances.size();  // 활성 인스턴스 수

    matches = Eigen::VectorXi::Zero(K);  // 매칭 결과 초기화
    if (M < 1) return 0;  // 활성 인스턴스가 없으면 0 반환

    // IoU 및 매칭 행렬 초기화
    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(K, M);
    Eigen::MatrixXi assignment = Eigen::MatrixXi::Zero(K, M);
    Eigen::MatrixXi assignment_colwise = Eigen::MatrixXi::Zero(K, M);
    Eigen::MatrixXi assignment_rowise = Eigen::MatrixXi::Zero(K, M);

    // IoU 계산
    for (int k_ = 0; k_ < K; k_++) {
        const auto &zk = detections[k_];  // 감지된 객체 k
        double zk_area = double(cv::countNonZero(zk->instances_idxs_));  // 감지된 객체의 영역 크기
        for (int m_ = 0; m_ < M; m_++) {
            auto instance_m = instance_map[active_instances[m_]];  // 활성 인스턴스 m
            cv::Mat overlap = instance_m->observed_image_mask->mul(zk->instances_idxs_);  // 겹치는 영역 계산
            double overlap_area = double(cv::countNonZero(overlap));  // 겹치는 영역 크기
            double instance_area = double(cv::countNonZero(*instance_m->observed_image_mask));  // 인스턴스 영역 크기
            iou(k_, m_) = overlap_area / (zk_area + instance_area - overlap_area);  // IoU 계산
        }
    }

    // 열 기준 최대 매칭 찾기
    for (int m_ = 0; m_ < M; m_++) {
        int max_row;
        double max_iou = iou.col(m_).maxCoeff(&max_row);  // 열에서 최대 IoU 및 행 인덱스 찾기
        if (max_iou > mapping_config.min_iou) assignment_colwise(max_row, m_) = 1;  // 최소 IoU 조건을 만족하면 매칭
    }

    // 행 기준 최대 매칭 찾기
    for (int k_ = 0; k_ < K; k_++) {
        int max_col;
        double max_iou = iou.row(k_).maxCoeff(&max_col);  // 행에서 최대 IoU 및 열 인덱스 찾기
        if (max_iou > mapping_config.min_iou) assignment_rowise(k_, max_col) = 1;  // 최소 IoU 조건을 만족하면 매칭

        // 모호한 매칭 쌍 탐지
        Eigen::ArrayXd row_correlated = iou.row(k_).array();  // 행의 IoU 값 배열화
        int row_correlated_num = (row_correlated > 0).count();  // 0보다 큰 IoU 값 개수
        if (row_correlated_num > 1) {  // 모호한 경우
            row_correlated[max_col] = 0.0;  // 최대 값 제외
            int second_max_col;
            row_correlated.maxCoeff(&second_max_col);  // 두 번째 최대 IoU 열 인덱스 찾기
            ambiguous_pairs.emplace_back(std::make_pair(active_instances[max_col], active_instances[second_max_col]));  // 모호한 쌍 추가
        }
    }

    assignment = assignment_colwise + assignment_rowise;  // 행렬 결합

    // 매칭 결과 도출
    int count = 0;
    std::vector<InstanceId> matched_instances;
    std::vector<InstanceId> unmatched_instances;
    for (int k_ = 0; k_ < K; k_++) {
        for (int m_ = 0; m_ < M; m_++) {
            if (assignment(k_, m_) == 2) {  // 양방향 매칭 확인
                matches(k_) = active_instances[m_];  // 매칭된 활성 인스턴스 ID 저장
                matched_instances.emplace_back(active_instances[m_]);
                count++;
                break;  // 매칭된 경우 다음 감지로 이동
            }
        }
    }

    // 매칭되지 않은 활성 인스턴스 식별
    for (int m_ = 0; m_ < M; m_++) {
        if (assignment.col(m_).sum() < 2) unmatched_instances.emplace_back(active_instances[m_]);  // 매칭되지 않은 인스턴스 추가
    }

    // 매칭 정보 로그 출력
    o3d_utility::LogInfo("{}/({},{}) associations out of detections and active instances.", count, K, M);

    return count;  // 총 매칭된 인스턴스 개수 반환
}


void SemanticMapping::refresh_all_semantic_dict()
{
    // 세맨틱 사전 서버 초기화
    semantic_dict_server.clear();

    // 모든 인스턴스를 순회하며 세맨틱 사전 업데이트
    for (auto instance : instance_map) {
        auto instance_labels = instance.second->get_predicted_class();  // 예측된 클래스 레이블 가져오기
        semantic_dict_server.update_instance(instance_labels.first, instance.first);  // 레이블과 인스턴스 ID를 업데이트
    }
}

int SemanticMapping::create_new_instance(const DetectionPtr &detection, const unsigned int &frame_id,
    const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose)
{
    // 새로운 인스턴스 생성
    auto instance = std::make_shared<Instance>(latest_created_instance_id + 1, frame_id, instance_config);

    // 프레임 데이터를 통합하여 새 인스턴스에 적용
    instance->integrate(frame_id, rgbd_image, pose.inverse());

    // 감지 정보를 기반으로 인스턴스 레이블 업데이트
    instance->update_label(detection);

    // 인스턴스 중심 업데이트
    instance->fast_update_centroid();

    // 인스턴스에 고유 색상 설정
    instance->color_ = InstanceColorBar20[instance->get_id() % InstanceColorBar20.size()];

    // 베이지안 라벨링이 활성화된 경우 확률 초기화 및 업데이트
    if (bayesian_label) {
        Eigen::VectorXf probability_vector;  // 확률 벡터 초기화
        instance->init_bayesian_fusion(bayesian_label->get_label_vec());  // 베이지안 융합 초기화
        bayesian_label->update_measurements(detection->labels_, probability_vector);  // 측정 업데이트
        instance->update_semantic_probability(probability_vector);  // 세맨틱 확률 업데이트
    }

    // 새 인스턴스를 인스턴스 맵에 추가
    instance_map.emplace(instance->get_id(), instance);

    // 최근 생성된 인스턴스 ID 업데이트
    latest_created_instance_id = instance->get_id();

    // 새로 생성된 인스턴스 ID 반환
    return instance->get_id();
}

bool SemanticMapping::IsSemanticSimilar(
    const std::unordered_map<std::string, float> &measured_labels_a,
    const std::unordered_map<std::string, float> &measured_labels_b)
{
    // 하나 이상의 레이블이 없는 경우 유사성 없음
    if (measured_labels_a.empty() || measured_labels_b.empty()) return false;

    // 두 레이블 맵에서 공통 레이블 확인
    for (const auto &label_score_a : measured_labels_a) {
        for (const auto &label_score_b : measured_labels_b) {
            if (label_score_a.first == label_score_b.first) return true;  // 공통 레이블 발견 시 true 반환
        }
    }
    return false;  // 공통 레이블이 없으면 false 반환
}

double SemanticMapping::Compute2DIoU(
    const open3d::geometry::OrientedBoundingBox &box_a, 
    const open3d::geometry::OrientedBoundingBox &box_b)
{
    // 두 박스를 Axis-Aligned Bounding Box(AABB)로 변환
    auto box_a_aligned = box_a.GetAxisAlignedBoundingBox();
    auto box_b_aligned = box_b.GetAxisAlignedBoundingBox();

    // 각 박스의 최소 및 최대 경계 추출
    Eigen::Vector3d a0 = box_a_aligned.GetMinBound();  // Box A의 최소 좌표
    Eigen::Vector3d a1 = box_a_aligned.GetMaxBound();  // Box A의 최대 좌표
    Eigen::Vector3d b0 = box_b_aligned.GetMinBound();  // Box B의 최소 좌표
    Eigen::Vector3d b1 = box_b_aligned.GetMaxBound();  // Box B의 최대 좌표

    // 겹치는 영역의 좌표 계산
    double x0 = std::max(a0(0), b0(0));  // 겹치는 영역의 최소 X 좌표
    double y0 = std::max(a0(1), b0(1));  // 겹치는 영역의 최소 Y 좌표
    double x1 = std::min(a1(0), b1(0));  // 겹치는 영역의 최대 X 좌표
    double y1 = std::min(a1(1), b1(1));  // 겹치는 영역의 최대 Y 좌표

    // 겹치는 영역이 없는 경우 IoU는 0
    if (x0 > x1 || y0 > y1) return 0.0;

    // 겹치는 영역 면적 계산
    double intersection_area = ((x1 - x0) * (y1 - y0));

    // 각 박스의 면적 계산
    double area_a = (a1(0) - a0(0)) * (a1(1) - a0(1));  // Box A의 면적
    double area_b = (b1(0) - b0(0)) * (b1(1) - b0(1));  // Box B의 면적

    // IoU 계산 (작은 값을 더해 분모가 0이 되는 것을 방지)
    double iou = intersection_area / (area_a + area_b - intersection_area + 1e-6);

    return iou;  // IoU 반환
}

double SemanticMapping::Compute3DIoU(
    const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation)
{
    // VoxelGrid를 사용하여 점 클라우드 A를 생성
    auto vxgrid_a = open3d::geometry::VoxelGrid::CreateFromPointCloud(
        *cloud_a, inflation * instance_config.voxel_length);

    // 점 클라우드 B의 점이 A에 포함되는지 확인
    std::vector<bool> overlap = vxgrid_a->CheckIfIncluded(cloud_b->points_);

    // IoU 계산
    double iou = double(std::count(overlap.begin(), overlap.end(), true)) /
                 double(overlap.size() + 0.000001);  // 분모에 작은 값을 추가하여 0으로 나눔 방지

    return iou;  // IoU 값 반환
}

int SemanticMapping::merge_overlap_instances(std::vector<InstanceId> instance_list)
{
    const double SEARCH_DISTANCE = 3.0;  // 검색 거리(미터 단위)
    std::vector<InstanceId> target_instances;

    // 병합 대상 인스턴스 목록 초기화
    if (instance_list.empty()) {
        for (const auto &instance_j : instance_map) {
            target_instances.emplace_back(instance_j.first);
        }
    } else {
        target_instances = instance_list;
    }

    // 병합할 대상 인스턴스가 3개 미만이면 종료
    if (target_instances.size() < 3) return 0;

    int old_instance_number = target_instances.size();  // 기존 인스턴스 수

    open3d::utility::Timer timer;  // 타이머 시작
    timer.Start();
    std::unordered_set<InstanceId> remove_instances;  // 병합될 인스턴스 ID 목록

    // 중첩 인스턴스 찾기 및 병합
    for (int i = 0; i < target_instances.size(); i++) {
        auto instance_i = instance_map[target_instances[i]];
        if (!instance_i->point_cloud) {
            o3d_utility::LogWarning("Instance {:d} has no point cloud", instance_i->get_id());
            continue;
        }

        // 점이 너무 적으면 병합하지 않음
        if (instance_i->point_cloud->points_.size() < 30) continue;

        for (int j = i + 1; j < target_instances.size(); j++) {
            if (remove_instances.find(target_instances[j]) != remove_instances.end()) continue;

            auto instance_j = instance_map[target_instances[j]];
            if (!instance_j->point_cloud) {
                o3d_utility::LogWarning("Instance {:d} has no point cloud", instance_j->get_id());
                continue;
            }

            // 점이 너무 적으면 병합하지 않음
            if (instance_j->point_cloud->points_.size() < 30) continue;

            // 중심 간 거리 계산
            double dist = (instance_i->centroid - instance_j->centroid).norm();

            // 세맨틱 유사성 확인 및 거리 조건 만족 여부 확인
            if (!IsSemanticSimilar(instance_i->get_measured_labels(), instance_j->get_measured_labels()) ||
                dist > SEARCH_DISTANCE) {
                continue;
            }

            // Spatial IoU 계산
            InstancePtr large_instance, small_instance;
            if (instance_i->point_cloud->points_.size() > instance_j->point_cloud->points_.size()) {
                large_instance = instance_i;
                small_instance = instance_j;
            } else {
                large_instance = instance_j;
                small_instance = instance_i;
            }

            double iou = Compute3DIoU(large_instance->point_cloud, small_instance->point_cloud,
                                      mapping_config.merge_inflation);

            // 병합 조건 만족 시 병합 수행
            if (iou > mapping_config.merge_iou) {
                large_instance->merge_with(
                    small_instance->get_complete_cloud(),
                    small_instance->get_measured_labels(),
                    small_instance->get_observation_count());

                if (bayesian_label) {
                    Eigen::VectorXf probability_vector;
                    bayesian_label->update_measurements(small_instance->get_measured_labels(), probability_vector);
                    large_instance->update_semantic_probability(probability_vector);
                }

                remove_instances.insert(small_instance->get_id());  // 병합된 인스턴스 추가
                if (small_instance->get_id() == instance_i->get_id()) break;  // 병합 완료 시 종료
            }
        }
    }

    // 병합된 인스턴스 제거
    for (auto &instance_id : remove_instances) {
        instance_map.erase(instance_id);
    }
    timer.Stop();  // 타이머 종료

    // 병합 결과 로그 출력
    std::cout << "Merged " << remove_instances.size() << "/" << old_instance_number
              << " instances by 3D IoU. It takes "
              << std::fixed << std::setprecision(1) << timer.GetDurationInMillisecond() << " ms.\n";

    return remove_instances.size();  // 병합된 인스턴스 수 반환
}

int SemanticMapping::merge_floor(bool verbose)
{
    // "floor"와 "carpet" 레이블의 인스턴스를 대상으로 설정
    std::vector<InstanceId> target_instances = semantic_dict_server.query_instances("floor");
    std::vector<InstanceId> carpet_instances = semantic_dict_server.query_instances("carpet");
    for (auto &carpet_id : carpet_instances) {
        target_instances.emplace_back(carpet_id);
    }

    // 병합 대상 인스턴스가 2개 미만인 경우 종료
    if (target_instances.size() < 2) return 0;
    if (verbose) std::cout << "Merging " << target_instances.size() << " floor instances\n";

    InstancePtr root_floor;  // 가장 큰 "floor" 인스턴스를 저장할 포인터
    Eigen::Vector3d root_center;  // root 인스턴스의 중심
    int root_points = 0;  // root 인스턴스의 포인트 수

    // 가장 큰 "floor" 인스턴스 찾기
    for (int i = 0; i < target_instances.size(); i++) {
        if (instance_map.find(target_instances[i]) == instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];

        if (instance->point_cloud->points_.size() > root_points) {
            root_floor = instance;
            root_center = instance->centroid;
            root_points = instance->point_cloud->points_.size();
        }
    }

    // root 인스턴스가 충분한 포인트를 가지고 있지 않으면 병합 종료
    if (root_points < 1000) return 0;
    if (verbose) std::cout << "Root floor has " << root_points << " points\n";

    // 병합 수행
    int count = 0;  // 병합된 인스턴스 수
    int debug = 0;  // 디버그용 확인한 인스턴스 수

    for (int i = 0; i < target_instances.size(); i++) {
        // root 인스턴스와 동일한 경우 제외
        if (target_instances[i] == root_floor->get_id()) continue;

        // 유효하지 않은 인스턴스 제외
        if (instance_map.find(target_instances[i]) == instance_map.end()) continue;
        auto instance = instance_map[target_instances[i]];

        // 포인트 수나 관측 횟수가 기준에 미치지 못하는 경우 제외
        if (instance->get_complete_cloud()->points_.size() < 500 ||
            instance->get_observation_count() < mapping_config.min_observation) {
            continue;
        }

        // Z축 기준 거리 계산
        double dist_z = (root_center - instance->centroid)[2];

        // Z축 거리 조건을 만족하는 경우 병합 수행
        if (dist_z < 1.0) {
            root_floor->merge_with(
                instance->get_complete_cloud(),
                instance->get_measured_labels(),
                instance->get_observation_count());

            if (bayesian_label) {
                Eigen::VectorXf probability_vector;
                bayesian_label->update_measurements(instance->get_measured_labels(),
                                                    probability_vector);
                root_floor->update_semantic_probability(probability_vector);
            }

            instance_map.erase(instance->get_id());  // 병합된 인스턴스 제거
            count++;  // 병합된 인스턴스 수 증가
        }

        debug++;  // 확인한 인스턴스 수 증가
    }

    if (verbose) std::cout << debug << " floor instances are checked\n";

    // 병합 결과 출력
    std::cout << "Merged " << count << " floor instances\n";
    return count;  // 병합된 인스턴스 수 반환
}


int SemanticMapping::merge_overlap_structural_instances(bool merge_all)
{
    // 현재 이 함수는 사용되지 않음
    assert(false);  // 사용 중단된 코드

    // "floor" 레이블을 가진 인스턴스 목록 생성
    std::vector<InstanceId> target_instances;
    for (auto &instance_j : instance_map) {
        if (instance_j.second->get_predicted_class().first == "floor")
            target_instances.emplace_back(instance_j.first);
    }

    // 병합 대상 인스턴스가 2개 미만이면 병합 수행하지 않음
    if (target_instances.size() < 2) return 0;

    // `merge_all`이 true인 경우 모든 floor 인스턴스를 하나의 가장 큰 인스턴스로 병합
    if (merge_all) {
        InstancePtr largest_floor;  // 가장 큰 floor 인스턴스를 저장할 포인터
        size_t largest_floor_size = 0;  // 가장 큰 floor 인스턴스의 포인트 개수

        // 가장 큰 floor 인스턴스 탐색
        for (auto idx : target_instances) {
            auto instance = instance_map[idx];
            if (instance->point_cloud->points_.size() > largest_floor_size) {
                largest_floor = instance;
                largest_floor_size = instance->point_cloud->points_.size();
            }
        }

        // 다른 모든 floor 인스턴스를 가장 큰 floor 인스턴스에 병합
        for (auto idx : target_instances) {
            if (idx == largest_floor->get_id()) continue;  // root 인스턴스는 건너뜀
            auto instance = instance_map[idx];
            largest_floor->merge_with(
                instance->point_cloud,
                instance->get_measured_labels(),
                instance->get_observation_count());
            instance_map.erase(idx);  // 병합된 인스턴스 제거
        }

        // 병합된 인스턴스 수 반환
        o3d_utility::LogInfo("Merged {:d} floor instances into one floor.", target_instances.size());
        return target_instances.size() - 1;
    }

    // "todo: remove" 섹션 - 미완성된 코드
    int old_instance_number = target_instances.size();
    std::unordered_set<InstanceId> remove_instances;

    // 모든 floor 인스턴스 쌍에 대해 중첩 여부 확인 및 병합 수행
    for (int i = 0; i < target_instances.size(); i++) {
        auto instance_i = instance_map[target_instances[i]];
        std::string label_i = instance_i->get_predicted_class().first;

        for (int j = i + 1; j < target_instances.size(); j++) {
            auto instance_j = instance_map[target_instances[j]];

            // 두 인스턴스의 2D IoU 계산
            InstancePtr large_instance, small_instance;
            if (instance_i->point_cloud->points_.size() > instance_j->point_cloud->points_.size()) {
                large_instance = instance_i;
                small_instance = instance_j;
            } else {
                large_instance = instance_j;
                small_instance = instance_i;
            }

            double iou = Compute2DIoU(*large_instance->min_box, *small_instance->min_box);

            // IoU가 기준값을 초과하는 경우 병합 수행
            if (iou > 0.03) {
                large_instance->merge_with(
                    small_instance->point_cloud,
                    small_instance->get_measured_labels(),
                    small_instance->get_observation_count());
                remove_instances.insert(small_instance->get_id());  // 병합된 인스턴스 기록

                // 현재 인스턴스가 병합된 경우 루프 종료
                if (small_instance->get_id() == instance_i->get_id()) break;
            }
        }
    }

    // 병합된 인스턴스 제거
    for (auto &instance_id : remove_instances) {
        instance_map.erase(instance_id);
    }

    // 병합된 인스턴스 수 반환 (이 섹션에서는 반환값이 명확하지 않음)
    return remove_instances.size();  // 병합된 인스턴스 수 반환
}

int SemanticMapping::merge_ambiguous_instances(const std::vector<std::pair<InstanceId, InstanceId>> &ambiguous_pairs)
{
    int count = 0;  // 병합된 인스턴스 수

    // 모호한 인스턴스 쌍을 순회하며 병합 시도
    for (const auto &pair : ambiguous_pairs) {
        auto instance_i = instance_map[pair.first];  // 첫 번째 인스턴스
        auto instance_j = instance_map[pair.second];  // 두 번째 인스턴스

        // 두 인스턴스 모두 점 클라우드를 가지고 있는 경우
        if (instance_i->point_cloud && instance_j->point_cloud) {
            // continue; 이 구문이 병합을 막고 있음. 이 코드는 제거해야 합니다.
            double iou = Compute3DIoU(instance_i->point_cloud, instance_j->point_cloud);  // 3D IoU 계산

            // IoU 기준에 따라 병합 수행
            if (iou > mapping_config.merge_iou) {
                instance_i->merge_with(
                    instance_j->point_cloud,
                    instance_j->get_measured_labels(),
                    instance_j->get_observation_count());

                // 베이지안 라벨링 업데이트
                if (bayesian_label) {
                    Eigen::VectorXf probability_vector;
                    bayesian_label->update_measurements(instance_j->get_measured_labels(), probability_vector);
                    instance_i->update_semantic_probability(probability_vector);
                }

                instance_map.erase(pair.second);  // 병합된 인스턴스 제거
                count++;  // 병합된 인스턴스 수 증가
            }
        }
        // 두 인스턴스 중 하나만 점 클라우드를 가지고 있는 경우
        else {
            O3d_Cloud_Ptr cloud_ptr;

            if (instance_i->point_cloud) {
                cloud_ptr = instance_i->point_cloud;
            } else if (instance_j->point_cloud) {
                cloud_ptr = instance_j->point_cloud;
            } else {
                continue;  // 점 클라우드가 없는 경우 처리 불가
            }

            // 병합 로직이 명확하지 않아 이 섹션은 추가 정의 필요
            // cloud_ptr을 활용한 병합 로직을 추가 가능
        }
    }

    // 병합 결과 출력
    o3d_utility::LogInfo("Merged {:d} ambiguous instances by 3D IoU.", count);
    return count;  // 병합된 인스턴스 수 반환
}

void SemanticMapping::extract_bounding_boxes()
{
    // 타이머 시작
    open3d::utility::Timer timer;
    timer.Start();

    int count = 0;  // 유효한 바운딩 박스 개수
    std::cout << "Extract bounding boxes for " << instance_map.size() << " instances\n";

    // 모든 인스턴스에 대해 바운딩 박스 추출
    for (const auto &instance : instance_map) {
        instance.second->filter_pointcloud_statistic();  // 통계적 필터링 수행

        // 포인트 개수가 최소 조건을 만족하는 경우에만 바운딩 박스 생성
        if (instance.second->get_cloud_size() > mapping_config.shape_min_points) {
            instance.second->CreateMinimalBoundingBox();  // 최소 바운딩 박스 생성
            count++;
        }
    }

    // 타이머 종료 및 결과 출력
    timer.Stop();
    o3d_utility::LogInfo("Extracted {:d} valid bounding boxes in {:f} ms", count, timer.GetDurationInMillisecond());
}

std::shared_ptr<open3d::geometry::PointCloud> SemanticMapping::export_global_pcd(bool filter, float vx_size)
{
    // 전역 포인트 클라우드 객체 초기화
    auto global_pcd = std::make_shared<open3d::geometry::PointCloud>();

    // 모든 인스턴스의 포인트 클라우드를 전역 클라우드에 병합
    for (const auto &inst : instance_map) {
        // 필터가 활성화된 경우 최소 포인트 조건을 만족하는 인스턴스만 포함
        if (filter && inst.second->get_cloud_size() < mapping_config.shape_min_points) continue;

        // 인스턴스의 전체 클라우드를 전역 클라우드에 추가
        *global_pcd += *inst.second->get_complete_cloud();
    }

    // Voxel 크기가 지정된 경우 다운샘플링 수행
    if (vx_size > 0.0) {
        global_pcd = global_pcd->VoxelDownSample(vx_size);
    }

    return global_pcd;  // 전역 포인트 클라우드 반환
}

std::vector<Eigen::Vector3d> SemanticMapping::export_instance_centroids(int earliest_frame_id) const
{
    // 인스턴스 중심 좌표를 저장할 벡터
    std::vector<Eigen::Vector3d> centroids;
    std::stringstream msg;

    // 모든 인스턴스를 순회하며 조건에 맞는 중심 좌표를 수집
    for (const auto &inst : instance_map) {
        int tmp_idx = inst.second->frame_id_;
        if (inst.second->get_cloud_size() > mapping_config.shape_min_points && tmp_idx >= earliest_frame_id) {
            centroids.emplace_back(inst.second->centroid);  // 중심 좌표 추가
            msg << inst.second->frame_id_ << ",";  // 프레임 ID 기록
        }
    }

    // 수집된 중심 좌표 개수 로그 출력
    o3d_utility::LogInfo("{:d} instance centroids are exported.", centroids.size());
    return centroids;  // 중심 좌표 반환
}

std::vector<std::string> SemanticMapping::export_instance_annotations(int earliest_frame_id) const
{
    // 인스턴스 레이블(클래스명)을 저장할 벡터
    std::vector<std::string> annotations;

    // 모든 인스턴스를 순회하며 조건에 맞는 레이블을 수집
    for (const auto &inst : instance_map) {
        int tmp_idx = inst.second->frame_id_;
        if (inst.second->get_cloud_size() > mapping_config.shape_min_points && tmp_idx >= earliest_frame_id) {
            annotations.emplace_back(inst.second->get_predicted_class().first);  // 클래스명 추가
        }
    }

    return annotations;  // 레이블 목록 반환
}

std::vector<std::shared_ptr<const open3d::geometry::Geometry>> SemanticMapping::get_geometries(bool point_cloud, bool bbox)
{
    // 시각화를 위한 Geometry 객체를 저장할 벡터
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> viz_geometries;

    // 모든 인스턴스를 순회하며 조건에 맞는 Geometry를 수집
    for (const auto &instance : instance_map) {
        if (instance.second->get_cloud_size() < mapping_config.shape_min_points) continue;  // 최소 포인트 조건 확인

        if (point_cloud) {
            viz_geometries.emplace_back(instance.second->point_cloud);  // 포인트 클라우드 추가
        }

        if (bbox && !instance.second->min_box->IsEmpty()) {
            viz_geometries.emplace_back(instance.second->min_box);  // 바운딩 박스 추가
        }
    }

    return viz_geometries;  // 수집된 Geometry 반환
}

void SemanticMapping::Transform(const Eigen::Matrix4d &pose)
{
    // 모든 인스턴스에 변환 행렬 적용
    for (const auto &instance : instance_map) {
        instance.second->point_cloud->Transform(pose);  // 포인트 클라우드 변환
        instance.second->centroid = instance.second->point_cloud->GetCenter();  // 중심 좌표 업데이트
    }
}

bool SemanticMapping::Save(const std::string &path)
{
    using namespace o3d_utility::filesystem;

    // 디렉토리가 존재하지 않으면 생성
    if (!DirectoryExists(path)) MakeDirectory(path);

    // 전역 포인트 클라우드 초기화
    open3d::geometry::PointCloud global_instances_pcd;

    // 인스턴스 정보와 바운딩 박스 정보를 저장할 벡터
    typedef std::pair<InstanceId, std::string> InstanceInfo;
    std::vector<InstanceInfo> instance_info;
    std::vector<std::string> instance_box_info;

    // 모든 인스턴스를 순회하며 저장 작업 수행
    for (const auto &instance : instance_map) {
        if (!instance.second->point_cloud) continue;  // 유효하지 않은 점 클라우드 무시

        LabelScore semantic_class_score = instance.second->get_predicted_class();  // 클래스 정보
        auto instance_cloud = instance.second->get_complete_cloud();  // 전체 클라우드 가져오기

        // 포인트 개수가 최소 기준 미만인 경우 무시
        if (instance.second->get_cloud_size() < mapping_config.shape_min_points) continue;

        // 전역 포인트 클라우드에 병합
        global_instances_pcd += *instance_cloud;

        // 인스턴스 정보 문자열 생성
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << instance.second->get_id();  // 인스턴스 ID
        open3d::io::WritePointCloud(path + "/" + ss.str() + ".ply", *instance_cloud);  // PLY 파일로 저장

        ss << ";"
           << semantic_class_score.first << "(" << std::fixed << std::setprecision(2) << semantic_class_score.second << ");"
           << instance.second->get_observation_count() << ";"
           << instance.second->get_measured_labels_string() << ";"
           << instance_cloud->points_.size() << ";\n";
        instance_info.emplace_back(instance.second->get_id(), ss.str());  // 정보 저장

        // 바운딩 박스 정보 생성
        if (!instance.second->min_box->IsEmpty()) {
            std::stringstream box_ss;
            box_ss << std::setw(4) << std::setfill('0') << instance.second->get_id() << ";";
            auto box = instance.second->min_box;
            box_ss << box->center_(0) << "," << box->center_(1) << "," << box->center_(2) << ";"
                   << box->R_.coeff(0, 0) << "," << box->R_.coeff(0, 1) << "," << box->R_.coeff(0, 2) << ","
                   << box->R_.coeff(1, 0) << "," << box->R_.coeff(1, 1) << "," << box->R_.coeff(1, 2) << ","
                   << box->R_.coeff(2, 0) << "," << box->R_.coeff(2, 1) << "," << box->R_.coeff(2, 2) << ";"
                   << box->extent_(0) << "," << box->extent_(1) << "," << box->extent_(2) << ";\n";
            instance_box_info.emplace_back(box_ss.str());  // 바운딩 박스 정보 저장
        }

        o3d_utility::LogInfo("Instance {:s} has {:d} points", semantic_class_score.first, instance_cloud->points_.size());
    }

    // 인스턴스 정보 정렬 및 저장
    std::sort(instance_info.begin(), instance_info.end(), [](const InstanceInfo &a, const InstanceInfo &b) {
        return a.first < b.first;
    });
    std::ofstream ofs(path + "/instance_info.txt", std::ofstream::out);
    ofs << "# instance_id;semantic_class(aggregate_score);observation_count;label_measurements;points_number\n";
    for (const auto &info : instance_info) {
        ofs << info.second;
    }
    ofs.close();

    // 바운딩 박스 정보 정렬 및 저장
    std::sort(instance_box_info.begin(), instance_box_info.end(), [](const std::string &a, const std::string &b) {
        return std::stoi(a.substr(0, 4)) < std::stoi(b.substr(0, 4));
    });
    std::ofstream ofs_box(path + "/instance_box.txt", std::ofstream::out);
    ofs_box << "# instance_id;center_x,center_y,center_z;R00,R01,R02,R10,R11,R12,R20,R21,R22;extent_x,extent_y,extent_z\n";
    for (const auto &info : instance_box_info) {
        ofs_box << info;
    }
    ofs_box.close();

    // 전역 인스턴스 맵 저장
    if (global_instances_pcd.points_.size() < 1) return false;
    open3d::io::WritePointCloud(path + "/instance_map.ply", global_instances_pcd);

    o3d_utility::LogWarning("Saved {} semantic instances to {:s}", instance_info.size(), path);
    return true;
}

bool SemanticMapping::load(const std::string &path)
{
    // SceneGraph 데이터를 지정된 경로에서 로드
    o3d_utility::LogInfo("Load SceneGraph from {:s}", path);
    using namespace o3d_utility::filesystem;

    // 경로에 디렉토리가 없으면 false 반환
    if (!DirectoryExists(path)) return false;

    // 인스턴스 정보 로드
    std::ifstream ifs(path + "/instance_info.txt", std::ifstream::in);
    std::string line;

    // 헤더 라인 건너뜀
    std::getline(ifs, line);

    // 각 인스턴스 정보 처리
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string instance_id_str;
        
        // 인스턴스 ID 추출
        std::getline(ss, instance_id_str, ';');
        InstanceId instance_id = std::stoi(instance_id_str);

        // 다른 정보 추출
        std::string label_score_str, observ_str, label_measurments_str, observation_count_str;
        std::getline(ss, label_score_str, ';');
        std::getline(ss, observ_str, ';');
        std::getline(ss, label_measurments_str, ';');

        // 새 인스턴스 생성 및 설정
        InstancePtr instance_toadd = std::make_shared<Instance>(instance_id, 10, instance_config);

        // 이전 레이블 정보 및 관측 횟수 로드
        instance_toadd->load_previous_labels(label_measurments_str);
        instance_toadd->load_obser_count(std::stoi(observ_str));

        // 포인트 클라우드 로드 및 중심 계산
        instance_toadd->point_cloud = open3d::io::CreatePointCloudFromFile(path + "/" + instance_id_str + ".ply");
        instance_toadd->centroid = instance_toadd->point_cloud->GetCenter();

        // 인스턴스 색상 설정
        instance_toadd->color_ = InstanceColorBar20[instance_id % InstanceColorBar20.size()];

        // 베이지안 라벨링 활성화 시 초기화 및 확률 업데이트
        if (bayesian_label) {
            Eigen::VectorXf probability_vector;
            instance_toadd->init_bayesian_fusion(bayesian_label->get_label_vec());
            bayesian_label->update_measurements(instance_toadd->get_measured_labels(), probability_vector);
            instance_toadd->update_semantic_probability(probability_vector);
        }

        // 인스턴스 맵에 추가
        instance_map.emplace(instance_id, instance_toadd);
    }

    // 로드된 인스턴스 수 로그 출력
    o3d_utility::LogInfo("Load {:d} instances", instance_map.size());

    return true;  // 성공적으로 로드되었음을 반환
}

void SemanticMapping::export_instances(
    std::vector<InstanceId> &names, std::vector<InstancePtr> &instances, int earliest_frame_id)
{
    // 로그 메시지 초기화
    std::stringstream msg;
    msg << "latest frames: ";

    // 모든 인스턴스를 순회하며 조건에 맞는 인스턴스 필터링
    for (auto &instance : instance_map) {
        if (!instance.second->point_cloud) continue;  // 포인트 클라우드가 없는 인스턴스 건너뜀
        msg << instance.second->frame_id_ << ",";  // 프레임 ID를 메시지에 추가

        // 포인트 개수와 프레임 ID 조건을 만족하는 경우 목록에 추가
        if (instance.second->get_cloud_size() > mapping_config.shape_min_points &&
            instance.second->frame_id_ > earliest_frame_id) {
            names.emplace_back(instance.first);  // 인스턴스 ID 추가
            instances.emplace_back(instance.second);  // 인스턴스 객체 추가
        }
    }

    msg << "\n";
    // 필터링 결과 로그 출력
    o3d_utility::LogInfo("{:s}", msg.str());
}

bool SemanticMapping::query_instance_info(const std::vector<InstanceId> &names,
                                          std::vector<Eigen::Vector3f> &centroids, 
                                          std::vector<std::string> &labels)
{
    // 지정된 인스턴스 ID 목록을 순회하며 정보 수집
    for (auto &name : names) {
        if (instance_map.find(name) == instance_map.end()) continue;  // 인스턴스가 없으면 건너뜀

        auto instance = instance_map[name];
        centroids.emplace_back(instance->centroid.cast<float>());  // 중심 좌표 추가
        labels.emplace_back(instance->get_predicted_class().first);  // 클래스 이름 추가
    }

    // 수집된 정보가 없는 경우 false 반환
    return !centroids.empty();
}

int SemanticMapping::merge_other_instances(std::vector<InstancePtr> &instances)
{
    int count = 0;  // 병합된 인스턴스 수

    // 주어진 인스턴스 목록을 순회하며 병합 수행
    for (auto &instance : instances) {
        // 최소 포인트 조건을 만족하지 않으면 건너뜀
        if (instance->point_cloud->points_.size() < mapping_config.shape_min_points) continue;

        // 인스턴스 ID 변경 (새로 생성된 ID로 초기화)
        instance->change_id(latest_created_instance_id + 1);

        // 인스턴스 맵에 추가
        instance_map.emplace(instance->get_id(), instance);
        latest_created_instance_id = instance->get_id();  // 최신 생성된 ID 갱신
        count++;
    }

    // 병합 결과 로그 출력
    o3d_utility::LogInfo("Merge {:d} instances", count);
    return count;  // 병합된 인스턴스 수 반환
}

} // namespace fmfusion

