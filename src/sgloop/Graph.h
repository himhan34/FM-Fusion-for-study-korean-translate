#ifndef FMFUSION_GRAPH_H
#define FMFUSION_GRAPH_H

#include <Common.h>  // 공통 라이브러리 포함
#include <mapping/Instance.h>  // 인스턴스 관련 정의 포함

namespace fmfusion
{

typedef std::array<uint32_t, 2> Corner;  // 두 개의 노드 ID를 저장하는 코너 타입 정의

// 노드를 정의하는 클래스
class Node
{
    public:
        // 생성자: 노드 ID와 인스턴스 ID로 초기화
        Node(uint32_t node_id_, InstanceId instance_id_) :
            id(node_id_), instance_id(instance_id_) {};

        /// @brief 코너를 샘플링하는 함수
        /// @param max_corner_number 최대 코너 개수
        /// @param corner_vector 결과 코너를 저장할 벡터
        /// @param padding_value 부족한 코너를 채우기 위한 패딩 값
        void sample_corners(const int &max_corner_number, std::vector<Corner> &corner_vector, int padding_value = 0);

        ~Node() {};

    public:
        uint32_t id;  // 노드 ID
        InstanceId instance_id;  // 원래 인스턴스 ID와 매칭
        std::string semantic;  // 노드의 의미적 레이블
        std::vector<uint32_t> neighbors;  // 이웃 노드 목록
        std::vector<Corner> corners;  // 코너 목록 (이웃 간 조합)
        O3d_Cloud_Ptr cloud;  // 노드의 포인트 클라우드
        Eigen::Vector3d centroid;  // 노드 중심 좌표
        Eigen::Vector3d bbox_shape;  // 바운딩 박스 크기 (x, y, z)
};
typedef std::shared_ptr<Node> NodePtr;  // Node 클래스의 스마트 포인터 정의

// 간선을 정의하는 클래스
class Edge
{
public:
    // 생성자: 시작 노드 ID와 끝 노드 ID로 초기화
    Edge(const uint32_t &id1, const uint32_t &id2) : src_id(id1), ref_id(id2) {
        distance = 0.0;
    }
    ~Edge() {};

public:
    uint32_t src_id;  // 시작 노드 ID
    uint32_t ref_id;  // 끝 노드 ID
    double distance;  // 간선의 거리
};
typedef std::shared_ptr<Edge> EdgePtr;  // Edge 클래스의 스마트 포인터 정의

// 그래프 데이터를 저장하는 구조체
struct DataDict {
    std::vector<Eigen::Vector3d> xyz;  // 포인트 클라우드
    std::vector<int> length_vec;  // 각 클라우드의 길이
    std::vector<uint32_t> labels;  // 노드 ID 레이블
    std::vector<Eigen::Vector3d> centroids;  // 중심 좌표
    std::vector<uint32_t> nodes;  // 노드 ID
    std::vector<uint32_t> instances;  // 인스턴스 ID

    // 인스턴스 ID를 출력하는 함수
    std::string print_instances() {
        std::stringstream msg;
        for (auto instance : instances) {
            msg << instance << ",";
        }
        return msg.str();
    }

    // 데이터 초기화
    void clear() {
        xyz.clear();
        length_vec.clear();
        labels.clear();
        centroids.clear();
        nodes.clear();
        instances.clear();
    }
};

// 그래프를 정의하는 클래스
class Graph
{
    public:
        Graph(GraphConfig config_);  // 생성자

        void initialize(const std::vector<InstancePtr> &instances);  // 그래프 초기화

        int subscribe_coarse_nodes(const float &latest_timestamp,
                                    const std::vector<uint32_t> &node_indices,
                                    const std::vector<uint32_t> &instances,
                                    const std::vector<Eigen::Vector3d> &centroids);  // 거친 노드 업데이트

        int subscribde_dense_points(const float &sub_timestamp,
                                     const std::vector<Eigen::Vector3d> &xyz,
                                     const std::vector<uint32_t> &labels);  // 밀집 포인트 클라우드 업데이트

        void construct_edges();  // 간선 생성
        void construct_triplets();  // 코너 생성

        const std::vector<NodePtr> get_const_nodes() const { return nodes; }  // 노드 반환
        const std::vector<EdgePtr> get_const_edges() const { return edges; }  // 간선 반환
        const std::vector<std::pair<int, int>> get_edges() const;  // 노드 간 간선 반환
        const std::vector<Eigen::Vector3d> get_centroids() const;  // 중심 좌표 반환

        void paint_all_floor(const Eigen::Vector3d &color);  // 바닥 노드에 색상 적용

        void extract_global_cloud(std::vector<Eigen::Vector3d> &xyz, std::vector<uint32_t> &labels);  // 전체 클라우드 추출

        DataDict extract_data_dict(bool coarse = false);  // 데이터 사전 추출
        O3d_Cloud_Ptr extract_global_cloud(float vx_size = -1.0) const;  // 다운샘플링된 클라우드 추출

        DataDict extract_coarse_data_dict();  // 간단한 데이터 사전 추출

        std::string print_nodes_names() const {
            std::stringstream ss;
            for (auto node : nodes) ss << node->instance_id << ",";
            return ss.str();
        }

        float get_timestamp() const { return timestamp; }  // 현재 타임스탬프 반환
        void clear();  // 그래프 초기화

        ~Graph() {};

    private:
        void update_neighbors();  // 이웃 노드 갱신

    public:
        int max_neighbor_number;  // 최대 이웃 수
        int max_corner_number;  // 최대 코너 수

    private:
        GraphConfig config;  // 그래프 설정
        std::map<InstanceId, int> instance2node_idx;  // 인스턴스 ID와 노드 인덱스 매핑
        std::vector<InstanceId> node_instance_idxs;  // 노드의 인스턴스 ID 목록
        std::vector<NodePtr> nodes;  // 노드 목록
        std::vector<EdgePtr> edges;  // 간선 목록

        int frame_id;  // 최근 프레임 ID
        float timestamp;  // 최근 타임스탬프
};
typedef std::shared_ptr<Graph> GraphPtr;  // Graph 클래스의 스마트 포인터 정의

}

#endif
