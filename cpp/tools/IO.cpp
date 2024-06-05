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
        msg<<match_pairs.size()<<" Matched pairs: \n";

        for (auto pair: match_pairs){
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            src_centroids.push_back(src_node->centroid);
            ref_centroids.push_back(ref_node->centroid);
            msg<<"("<<src_node->instance_id<<","<<ref_node->instance_id<<") "
            <<"("<<src_node->semantic<<","<<ref_node->semantic<<")\n";
        }

        // std::cout<<msg.str()<<std::endl;
    };

    bool save_match_results(const Eigen::Matrix4d &pose,
                            const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                            const std::vector<float> &match_scores,
                            const std::string &output_file_dir)
    {
        std::ofstream file(output_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_file_dir<<std::endl;
            return false;
        }

        file<<"# pose\n";
        file<<std::fixed<<std::setprecision(6);
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                file<<pose(i,j)<<" ";
            }
            file<<std::endl;
        }

        file<<"# src, ref, score\n";
        file<<std::fixed<<std::setprecision(3);
        for (size_t i=0; i<match_pairs.size(); i++){
            file<<"("<<match_pairs[i].first<<","<<match_pairs[i].second<<") "
            <<match_scores[i]<<std::endl;
        }

        file.close();
        return true;
    };

    
} // namespace name



}
