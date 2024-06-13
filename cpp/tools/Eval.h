#include <fstream>
#include <vector>
#include <chrono>
#include "Utility.h"

namespace fmfusion
{
    
    int maks_true_instance(const std::string &gt_file_dir, const std::vector<std::pair<uint32_t,uint32_t>> &pred_instances,
                            std::vector<bool> &pred_masks)
                            // std::string &msg)
    {
        std::cout<<"Loading gt instance file: "<<gt_file_dir<<std::endl;
        std::set<std::string> gt_instance_pairs;
        int M = pred_instances.size();
        int count_true = 0;

        // Load gt
        std::ifstream gt_file(gt_file_dir, std::ifstream::in);
        if (!gt_file.is_open()){
            std::cerr<<"Error opening file: "<<gt_file_dir<<std::endl;
            return -1;
        }

        std::string line;
        while (std::getline(gt_file, line)){
            // std::cout<<line<<std::endl;
            auto eles = utility::split_str(line, " ");
            std::stringstream pair_name;
            pair_name<<eles[0]<<"_"<<eles[1];
            // std::cout<<pair_name.str()<<std::endl;
            gt_instance_pairs.insert(pair_name.str());
        }

        //
        for (int i=0; i<M; i++){
            std::stringstream pair_name;
            pair_name<<pred_instances[i].first<<"_"<<pred_instances[i].second;
            bool true_mask=false;
            if(gt_instance_pairs.find(pair_name.str())!=gt_instance_pairs.end()){
                true_mask=true;
                count_true++;
            }
            pred_masks.push_back(true_mask);
            // std::cout<<pair_name.str()<<" "<<true_mask<<std::endl;
        }

        return count_true;
    }    


} // namespace fmfusion





