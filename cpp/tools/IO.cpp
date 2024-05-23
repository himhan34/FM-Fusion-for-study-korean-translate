#include "Tools.h"

namespace fmfusion
{

namespace IO
{
    void extract_match_instances(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            const std::vector<fmfusion::NodePtr> &src_nodes,
                            const std::vector<fmfusion::NodePtr> &ref_nodes,
                            std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> &match_instances)
    {

        for (auto pair: match_pairs){
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            match_instances.push_back(std::make_pair(src_node->instance_id, ref_node->instance_id));
        }

    }

    void extract_instance_correspondences(const std::vector<fmfusion::NodePtr> &src_nodes, 
                                            const std::vector<fmfusion::NodePtr> &ref_nodes, 
                                            const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                                            const std::vector<float> &match_scores,
                                            std::vector<Eigen::Vector3d> &src_centroids, std::vector<Eigen::Vector3d> &ref_centroids)
    {
        std::stringstream msg;
        msg<<match_pairs.size()<<"Matched pairs: \n";

        for (auto pair: match_pairs){
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            src_centroids.push_back(src_node->centroid);
            ref_centroids.push_back(ref_node->centroid);
            msg<<"("<<src_node->instance_id<<","<<ref_node->instance_id<<") "
            <<"("<<src_node->semantic<<","<<ref_node->semantic<<")\n";
        }

        std::cout<<msg.str()<<std::endl;
    };

    
} // namespace name



}
