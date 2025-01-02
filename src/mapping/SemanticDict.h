#ifndef SEMANTICDICT_H
#define SEMANTICDICT_H

#include <unordered_map> // 해시 맵을 사용하기 위한 헤더 파일

namespace fmfusion // fmfusion 네임스페이스 정의
{
    // SemanticDictionary 타입 정의: 문자열 키와 InstanceId 리스트를 값으로 가지는 해시 맵
    typedef std::unordered_map<std::string, InstanceIdList> SemanticDictionary;

    // SemanticDictServer 클래스 정의
    class SemanticDictServer
    {
    public:
        // 생성자: semantic_dict를 초기화하고 비우기
        SemanticDictServer():semantic_dict()
        {
            semantic_dict.clear(); // 딕셔너리 초기화
        };

        // 소멸자: 현재는 특별히 할 작업 없음
        ~SemanticDictServer(){};

        /// @brief 의미 체계를 업데이트하는 함수
        /// @param semantic_label 의미 라벨
        /// @param instance_id 인스턴스 ID
        void update_instance(const std::string &semantic_label, 
                            const InstanceId &instance_id)
        {
            // 의미 라벨이 "floor" 또는 "carpet"인 경우 처리
            if(semantic_label=="floor" || semantic_label=="carpet"){
                if(semantic_dict.find(semantic_label) == semantic_dict.end()) // 라벨이 없으면 새로 추가
                    semantic_dict[semantic_label] = {instance_id};
                else // 라벨이 이미 있으면 인스턴스 ID 추가
                    semantic_dict[semantic_label].push_back(instance_id);
            }
        };

        // semantic_dict를 비우는 함수
        void clear()
        {
            semantic_dict.clear();  
        };

        // 특정 라벨에 해당하는 인스턴스 ID 리스트를 반환하는 함수
        std::vector<InstanceId> query_instances(const std::string &label)
        {
            if(semantic_dict.find(label) == semantic_dict.end()) // 라벨이 없으면 빈 리스트 반환
                return {};
            return semantic_dict[label]; // 라벨이 있으면 해당 인스턴스 리스트 반환
        }

        // semantic_dict를 반환하는 함수 (주석 처리됨)
        // SemanticDictionary get_semantic_dict()
        // {
        //     return semantic_dict;
        // }

    private:
        SemanticDictionary semantic_dict; // 의미 체계 딕셔너리
    };

}

#endif // SEMANTICDICT_H
