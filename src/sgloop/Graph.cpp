#include "Graph.h"  // Graph 클래스와 Node 클래스 선언이 포함된 헤더 파일 포함

namespace fmfusion
{
    void Node::sample_corners(const int &max_corner_number, std::vector<Corner> &corner_vector, int padding_value)
    {
        // corners에서 샘플링한 결과를 corner_vector에 저장하는 함수
        // 최대 코너 개수(max_corner_number)에 맞게 코너를 샘플링하거나, 부족하면 패딩 값을 추가

        // 디버깅 메시지를 저장하기 위한 스트링 스트림 초기화
        std::stringstream msg;
        msg<<corners.size()<<" corners: ";  // 현재 코너 개수를 메시지에 추가
        corner_vector.reserve(max_corner_number);  // corner_vector의 용량을 미리 예약

        // 패딩에 사용될 Corner 객체 생성
        Corner padding_corner = {(uint32_t)padding_value, (uint32_t)padding_value};

        if (corners.size()<max_corner_number){
            // 현재 코너 개수가 최대 코너 개수보다 적은 경우

            corner_vector.insert(corner_vector.end(), corners.begin(), corners.end());  // 코너 벡터에 기존 코너 추가

            // 부족한 부분을 패딩 값으로 채움
            for (int i=corners.size(); i<max_corner_number; i++){
                corner_vector.emplace_back(padding_corner);
            }
        }
        else if (corners.size()==max_corner_number){
            // 현재 코너 개수가 최대 코너 개수와 같은 경우

            corner_vector.insert(corner_vector.end(), corners.begin(), corners.end());  // 그대로 복사
        }
        else{
            // 현재 코너 개수가 최대 코너 개수보다 많은 경우

            std::vector<int> indices(corners.size());  // 인덱스를 저장할 벡터 생성
            std::iota(indices.begin(), indices.end(), 0);  // 0부터 시작하는 연속된 값으로 채움
            std::random_shuffle(indices.begin(), indices.end());  // 인덱스를 무작위로 섞음
            for (int i=0; i<max_corner_number; i++){
                corner_vector.emplace_back(corners[indices[i]]);  // 무작위로 선택된 코너를 추가
            }
        }
        assert(corner_vector.size()==max_corner_number);  // corner_vector 크기 확인 (디버깅용)
        msg<<"\n";  // 메시지 끝에 줄바꿈 추가
        // std::cout<<msg.str();  // 디버깅 메시지 출력 (현재 비활성화)
    }

    Graph::Graph(GraphConfig config_):config(config_),max_corner_number(0),max_neighbor_number(0),frame_id(-1),timestamp(-1.0)
    {
        // Graph 클래스의 생성자
        // 기본 초기화 및 "GNN 초기화" 메시지 출력

        std::cout<<"GNN initialized.\n";  // 초기화 완료 메시지 출력
    }

    void Graph::initialize(const std::vector<InstancePtr> &instances)
    {
        // 그래프를 초기화하는 함수
        // 입력된 인스턴스 리스트를 기반으로 노드 생성 및 설정

        o3d_utility::Timer timer_;  // 실행 시간 측정을 위한 타이머 객체 생성
        timer_.Start();  // 타이머 시작
        frame_id = 1;  // 초기 프레임 ID 설정

        for (auto inst:instances){
            // 입력된 각 인스턴스를 처리하여 노드로 변환

            NodePtr node = std::make_shared<Node>(nodes.size(), inst->get_id());  // 새 노드 생성
            std::string label = inst->get_predicted_class().first;  // 예측된 클래스 레이블 가져오기
            if (config.ignore_labels.find(label)!=std::string::npos) continue;  // 무시할 레이블인 경우 건너뜀

            node->semantic = label;  // 노드의 레이블 설정
            node->centroid = inst->centroid;  // 중심 좌표 설정
            node->bbox_shape = inst->min_box->extent_;  // 바운딩 박스 크기 설정
            node->cloud = std::make_shared<open3d::geometry::PointCloud>(*inst->point_cloud);  // 포인트 클라우드 깊은 복사
            if (config.voxel_size>0.0)
                node->cloud = node->cloud->VoxelDownSample(config.voxel_size);  // 복셀 다운샘플링 수행

            nodes.push_back(node);  // 생성된 노드를 노드 리스트에 추가
            node_instance_idxs.push_back(inst->get_id());  // 인스턴스 ID를 노드 ID 리스트에 추가
            instance2node_idx[inst->get_id()] = node->id;  // 인스턴스 ID와 노드 ID 매핑
        }
        timer_.Stop();  // 타이머 종료
        std::cout<<"Constructed "<<nodes.size()<<" nodes in "
                        <<std::fixed<<std::setprecision(3)
                        <<timer_.GetDurationInMillisecond()<< " ms.\n";  // 노드 생성 완료 메시지 출력
    }

    int Graph::subscribe_coarse_nodes(const float &latest_timestamp,
                                        const std::vector<uint32_t> &node_indices,
                                        const std::vector<uint32_t> &instances,
                                        const std::vector<Eigen::Vector3d> &centroids)
    {
        // 최신 타임스탬프와 현재 타임스탬프 차이가 0.01 이상일 경우 노드를 초기화하고 갱신
        if(latest_timestamp - timestamp > 0.01){
            clear();  // 기존 그래프 데이터를 초기화
            std::stringstream msg;  // 디버깅 메시지를 저장하기 위한 스트링 스트림 초기화
            for (int i=0; i<instances.size(); i++){
                NodePtr node = std::make_shared<Node>(node_indices[i], instances[i]);  // 새로운 노드 생성
                node->centroid = centroids[i];  // 노드 중심 좌표 설정
                node->cloud = std::make_shared<open3d::geometry::PointCloud>();  // 빈 포인트 클라우드 초기화
                nodes.push_back(node);  // 생성된 노드를 노드 리스트에 추가
                node_instance_idxs.push_back(instances[i]);  // 인스턴스 ID를 노드 ID 리스트에 추가
                instance2node_idx[instances[i]] = i;  // 인스턴스 ID와 노드 ID 매핑
                msg<<instances[i]<<",";  // 디버깅 메시지에 인스턴스 ID 추가
            }
            timestamp = latest_timestamp;  // 그래프 타임스탬프 갱신

            return instances.size();  // 추가된 인스턴스 수 반환
        }
        else return 0;  // 타임스탬프 차이가 작을 경우 노드 갱신하지 않고 0 반환
    }

    int Graph::subscribde_dense_points(const float &sub_timestamp,
                                        const std::vector<Eigen::Vector3d> &xyz,
                                        const std::vector<uint32_t> &labels)
    {
        // 밀집 포인트 데이터를 그래프에 추가
        int count = 0; // 갱신된 포인트 수를 저장할 변수
        int N = nodes.size();  // 현재 그래프에 있는 노드 수
        int X = xyz.size();  // 입력된 포인트 데이터 수
        std::vector<std::vector<Eigen::Vector3d>> nodes_points(N, std::vector<Eigen::Vector3d>());  // 각 노드에 속하는 포인트를 저장할 벡터 초기화
        std::stringstream msg;  // 디버깅 메시지를 저장하기 위한 스트링 스트림 초기화

        for(int k=0;k<X;k++){
            // 각 포인트의 레이블에 따라 노드에 할당
            if (labels[k]>=N){
                std::cout<<"Node "<<labels[k]<<" not found in the graph.\n";  // 노드가 그래프에 없는 경우 경고 메시지 출력
                continue;  // 다음 포인트로 건너뜀
            }
            nodes_points[labels[k]].push_back(xyz[k]);  // 해당 레이블의 노드에 포인트 추가
        }

        for(int i=0;i<N;i++){
            // 노드에 새롭게 추가된 포인트가 있는 경우
            if (nodes_points[i].empty()) continue;  // 포인트가 없는 경우 건너뜀
            nodes[i]->cloud = std::make_shared<open3d::geometry::PointCloud>(nodes_points[i]);  // 포인트 클라우드 갱신
            count += nodes_points[i].size();  // 갱신된 포인트 수 누적
        }

        return count;  // 총 갱신된 포인트 수 반환
    }

    void Graph::construct_edges()
    {
        // 그래프에서 노드 간의 간선을 생성하는 함수
        // 노드들 사이의 거리 및 특정 조건에 따라 간선을 추가

        std::string floor_names = "floor. carpet.";  // 바닥과 관련된 레이블 정의
        std::string ceiling_names = "ceiling.";  // 천장과 관련된 레이블 정의
        std::set<uint32_t> floors, ceilings;  // 바닥 및 천장 노드의 인덱스를 저장
        const int N = nodes.size();  // 그래프의 노드 개수
        std::stringstream msg;  // 디버깅 메시지를 저장하기 위한 스트링 스트림 초기화
        std::stringstream edge_msgs;  // 간선 정보 메시지 저장
        float MIN_SEARCH_RADIUS = 1.0;  // 최소 탐색 반경
        float MAX_SEARCH_RADIUS = 6.0;  // 최대 탐색 반경
        o3d_utility::Timer timer_;  // 실행 시간 측정을 위한 타이머 객체 생성
        timer_.Start();  // 타이머 시작

        // 노드 간 객체-객체 연결
        for (int i=0; i<N; i++){
            const NodePtr src = nodes[i];  // 현재 노드 가져오기
            float radius_src;

            // 노드가 바닥 또는 천장 레이블에 속하는 경우 인덱스를 저장하고 건너뜀
            if (floor_names.find(src->semantic) != std::string::npos){
                floors.emplace(i);
                continue;
            }
            if (ceiling_names.find(src->semantic) != std::string::npos){
                ceilings.emplace(i);
                continue;
            }

            // 벽인 경우 최소 탐색 반경 사용, 아니면 바운딩 박스 크기를 기준으로 반경 계산
            if (src->semantic.find("wall") != std::string::npos)
                radius_src = MIN_SEARCH_RADIUS;
            else
                radius_src = src->bbox_shape.norm() / 2.0;

            msg << src->semantic << "(" << radius_src << "): ";

            for (int j = i + 1; j < N; j++) {
                const NodePtr ref = nodes[j];  // 비교할 다른 노드 가져오기

                // 동일 노드이거나 바닥/천장 노드인 경우 건너뜀
                if (src->id == ref->id || 
                    floor_names.find(ref->semantic) != std::string::npos || 
                    ceiling_names.find(ref->semantic) != std::string::npos)
                    continue;

                float radius_ref;

                // 비교 대상이 벽인 경우 최소 탐색 반경 사용, 아니면 바운딩 박스 크기를 기준으로 반경 계산
                if (ref->semantic.find("wall") != std::string::npos)
                    radius_ref = MIN_SEARCH_RADIUS;
                else
                    radius_ref = ref->bbox_shape.norm() / 2.0;

                // 두 노드 간의 탐색 반경 계산
                float search_radius = config.edge_radius_ratio * std::max(radius_src, radius_ref);
                search_radius = std::max(std::min(search_radius, MAX_SEARCH_RADIUS), MIN_SEARCH_RADIUS);

                // 두 노드 간의 거리 계산
                float dist = (src->centroid - ref->centroid).norm();

                msg << "(" << radius_ref << ")(" << search_radius << "),";

                // 두 노드 간의 거리가 탐색 반경 내에 있으면 간선 추가
                if (dist < search_radius) {
                    EdgePtr edge = std::make_shared<Edge>(i, j);  // 간선 생성
                    edges.push_back(edge);  // 간선 리스트에 추가
                    edge_msgs << "(" << src->instance_id << "," << ref->instance_id << "),";
                }
            }
            msg << "\n";
        }

        // 바닥과의 연결을 포함하도록 설정된 경우
        if (config.involve_floor_edge && !floors.empty()) {
            for (int i = 0; i < N; i++) {  // 각 노드를 가장 가까운 바닥 노드와 연결
                if (floors.find(i) != floors.end()) continue;  // 이미 바닥 노드인 경우 건너뜀

                std::pair<int, float> closet_floor = std::make_pair(-1, 1000000.0);  // 가장 가까운 바닥 노드 초기화
                const NodePtr src = nodes[i];

                for (auto floor_index : floors) {
                    float dist = (src->centroid - nodes[floor_index]->centroid).norm();  // 거리 계산
                    if (dist < closet_floor.second)
                        closet_floor = std::make_pair(floor_index, dist);  // 더 가까운 노드로 갱신
                }

                if (closet_floor.first >= 0) {  // 유효한 간선인 경우 추가
                    EdgePtr edge = std::make_shared<Edge>(i, closet_floor.first);
                    edges.push_back(edge);
                }
            }
        }

        timer_.Stop();  // 타이머 종료

        // 간선 생성 결과 업데이트
        update_neighbors();
    }

    void Graph::update_neighbors()
    {
        // 간선 정보를 기반으로 각 노드의 이웃 노드 목록을 업데이트
        for (auto edge : edges) {
            nodes[edge->src_id]->neighbors.push_back(edge->ref_id);  // 간선의 시작 노드에 이웃 추가
            nodes[edge->ref_id]->neighbors.push_back(edge->src_id);  // 간선의 끝 노드에 이웃 추가
        }
    }

    void Graph::construct_triplets()
    {
        // 각 노드의 이웃 간 조합으로 코너(Triplet)를 생성
        for (NodePtr &node : nodes) {
            std::stringstream neighbor_msg, corner_msg;  // 디버깅 메시지 초기화
            if (node->neighbors.size() < 2) continue;  // 이웃이 2개 미만이면 건너뜀

            for (int i = 0; i < node->neighbors.size(); i++) {
                neighbor_msg << node->neighbors[i] << ",";
                for (int j = i + 1; j < node->neighbors.size(); j++) {
                    Corner corner = {node->neighbors[i], node->neighbors[j]};  // 코너 생성
                    node->corners.push_back(corner);  // 노드에 코너 추가
                    corner_msg << "(" << corner[0] << "," << corner[1] << "),";
                }
            }

            // 최대 이웃 수와 최대 코너 수 업데이트
            if (node->neighbors.size() > max_neighbor_number) max_neighbor_number = node->neighbors.size();
            if (node->corners.size() > max_corner_number) max_corner_number = node->corners.size();
        }
        assert(max_corner_number > 0);  // 코너가 존재해야 함
    }

    void Graph::extract_global_cloud(std::vector<Eigen::Vector3d> &xyz, std::vector<uint32_t> &labels)
    {
        // 그래프의 모든 노드에서 포인트 클라우드와 라벨을 추출
        std::stringstream msg;
        msg << "Nodes id: ";
        for (auto node : nodes) {
            xyz.insert(xyz.end(), node->cloud->points_.begin(), node->cloud->points_.end());  // 포인트 추가
            labels.insert(labels.end(), node->cloud->points_.size(), node->id);  // 라벨 추가
            msg << node->id << ",";
        }
    }

    O3d_Cloud_Ptr Graph::extract_global_cloud(float vx_size) const
    {
        // 모든 노드의 포인트 클라우드를 합쳐서 반환
        O3d_Cloud_Ptr out_cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
        for (auto node : nodes) {
            *out_cloud_ptr += *(node->cloud);  // 노드의 클라우드 병합
        }
        if (vx_size > 0.0) out_cloud_ptr->VoxelDownSample(vx_size);  // 다운샘플링 수행
        return out_cloud_ptr;
    }

    void Graph::clear()
    {
        // 그래프의 모든 데이터를 초기화
        nodes.clear();
        edges.clear();
        node_instance_idxs.clear();
        instance2node_idx.clear();
        max_corner_number = 0;
        max_neighbor_number = 0;
    }

    DataDict Graph::extract_data_dict(bool coarse)
    {
        // 그래프 데이터를 DataDict 형식으로 추출
        DataDict data_dict;
        for (auto node : nodes) {
            data_dict.centroids.push_back(node->centroid);  // 중심 좌표 추가
            data_dict.nodes.push_back(node->id);  // 노드 ID 추가
            data_dict.instances.push_back(node->instance_id);  // 인스턴스 ID 추가
            if (!coarse && node->cloud.use_count() > 0) {
                data_dict.xyz.insert(data_dict.xyz.end(), node->cloud->points_.begin(), node->cloud->points_.end());  // 포인트 추가
                data_dict.labels.insert(data_dict.labels.end(), node->cloud->points_.size(), node->id);  // 라벨 추가
            }
        }
        data_dict.length_vec = std::vector<int>(1, data_dict.xyz.size());  // 길이 벡터 설정
        return data_dict;
    }

    void Graph::paint_all_floor(const Eigen::Vector3d &color)
    {
        // 바닥 레이블에 해당하는 노드의 포인트 클라우드를 지정된 색상으로 칠함
        std::string floor_names = "floor. carpet.";
        for (auto node : nodes) {
            if (floor_names.find(node->semantic) != std::string::npos) {
                node->cloud->PaintUniformColor(color);  // 포인트 클라우드에 색상 적용
            }
        }
    }

    const std::vector<Eigen::Vector3d> Graph::get_centroids() const
    {
        // 그래프의 모든 노드의 중심 좌표를 반환
        std::vector<Eigen::Vector3d> centroids;
        for (auto node : nodes) {
            centroids.push_back(node->centroid);
        }
        return centroids;
    }

    const std::vector<std::pair<int, int>> Graph::get_edges() const
    {
        // 그래프의 모든 간선을 반환
        std::vector<std::pair<int, int>> edge_pairs;
        for (auto edge : edges) {
            edge_pairs.push_back(std::make_pair(edge->src_id, edge->ref_id));
        }
        return edge_pairs;
    }


