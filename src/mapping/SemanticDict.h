#ifndef SEMANTICDICT_H
#define SEMANTICDICT_H

#include <unordered_map>

namespace fmfusion
{
    typedef unordered_map<std::string, InstanceIdList> SemanticDictionary;

    class SemanticDictServer
    {
    public:
        SemanticDictServer():semantic_dict()
        {
            semantic_dict.clear();
        };

        ~SemanticDictServer(){};

        /// @brief  Update the semantic dictionary.
        /// @param semantic_label 
        /// @param instance_id 
        void update_instance(const std::string &semantic_label, 
                            const InstanceId &instance_id)
        {
            if(semantic_label=="floor" ||semantic_label=="carpet"){
                if(semantic_dict.find(semantic_label)==semantic_dict.end())
                    semantic_dict[semantic_label] = {instance_id};
                else
                    semantic_dict[semantic_label].push_back(instance_id);
            }
        };

        void clear()
        {
            semantic_dict.clear();  
        };

        std::vector<InstanceId> query_instances(const std::string &label)
        {
            if(semantic_dict.find(label)==semantic_dict.end())
                return {};
            return semantic_dict[label];
        }

        // SemanticDictionary get_semantic_dict()
        // {
        //     return semantic_dict;
        // }

    private:
        SemanticDictionary semantic_dict;
    };

}

#endif
