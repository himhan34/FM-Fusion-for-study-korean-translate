#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <mapping/SemanticMapping.h>  // SemanticMapping 헤더 파일 포함
#include <sgloop/Graph.h>  // Graph 헤더 파일 포함

namespace fmfusion
{

    // 씬 그래프 초기화 함수
    bool init_scene_graph(const Config &config, const std::string &scene_dir,
                          std::shared_ptr<Graph> &src_graph)
    {
        std::vector<InstanceId> instance_idxs;  // 인스턴스 ID 목록
        std::vector<InstancePtr> instances;  // 인스턴스 포인터 목록

        // SemanticMapping 객체 생성
        auto src_map = std::make_shared<SemanticMapping>(
            SemanticMapping(config.mapping_cfg, config.instance_cfg));
        
        // Graph 객체 생성
        src_graph = std::make_shared<Graph>(config.graph);

        // 맵 준비
        src_map->load(scene_dir);  // 씬 디렉토리에서 맵 로드
        src_map->extract_bounding_boxes();  // 경계 상자 추출
        src_map->export_instances(instance_idxs, instances);  // 인스턴스 추출
        assert(instance_idxs.size() > 0 && instances.size() > 0);  // 인스턴스가 비어있지 않도록 확인

        // 그래프 준비
        src_graph->initialize(instances);  // 인스턴스로 그래프 초기화
        src_graph->construct_edges();  // 그래프의 엣지 생성
        src_graph->construct_triplets();  // 그래프의 삼중항 생성
        // fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();  // 주석 처리된 코드

        return true;  // 성공적으로 초기화 완료
    }

}

#endif // INITIALIZATION_H
