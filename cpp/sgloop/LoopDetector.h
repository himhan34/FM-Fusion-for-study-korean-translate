#ifndef LOOP_DETECTOR_H_
#define LOOP_DETECTOR_H_
#include <Common.h>
#include <sgloop/SGNet.h>
#include <sgloop/Graph.h>
#include <sgloop/ShapeEncoder.h>


namespace fmfusion
{
    struct ImplicitGraph{
        torch::Tensor node_features; // (N, D0)
        torch::Tensor shape_features; // (N, D1)
        torch::Tensor node_knn_points; // (N, K, 3)
        torch::Tensor node_knn_features; // (N, K, D2)
        bool shape_embedded = false;
    };

    class LoopDetector
    {
    public:
        LoopDetector(LoopDetectorConfig &lcd_config,
                    ShapeEncoderConfig &shape_encoder_config, 
                    SgNetConfig &sgnet_config, 
                    const std::string weight_folder,
                    int cuda_number=0,
                    std::vector<std::string> ref_graph_names={});
        ~LoopDetector() {};

        /// \brief  Encode the reference scene graph. 
        /// If \param data_dict is empty, it only run triplet gnn and skip dense point cloud encoding.
        bool encode_ref_scene_graph(const std::string &ref_name, const std::vector<NodePtr> &nodes);

        /// \brief  Encode the source scene graph. 
        /// If \param data_dict is empty, it only run triplet gnn and skip dense point cloud encoding.
        bool encode_src_scene_graph(const std::vector<NodePtr> &nodes);

        /// \brief Update the ref sg features from subscribed com server.
        bool subscribe_ref_coarse_features(const std::string &ref_name,
                                        const float &cur_timestamp, 
                                        const std::vector<std::vector<float>> &coarse_features_vec,
                                        torch::Tensor coarse_features);

        /// \brief the point cloud and instance points from two scene graphs to encode.
        /// @param hidden_feat_dir: the directory to save the input sent to shape encoder.
        bool encode_concat_sgs(const std::string &ref_name, 
                                const int &Nr, const DataDict& ref_data_dict,
                                const int& Ns, const DataDict& src_data_dict,
                                float &encoding_time,
                                bool fused=false,
                                std::string hidden_feat_dir="");

        int match_nodes(const std::string &ref_name,
                        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                        std::vector<float> &match_scores,
                        bool fused=false,
                        std::string dir="");

        /// \brief  Match the points of the corresponding instances.
        /// \param  match_pairs         The matched node pairs.
        /// \param  corr_src_points     (C,3) The corresponding points in the source instance.
        /// \param  corr_ref_points     (C,3) The corresponding points in the reference instance.
        /// \param  corr_match_indices  (C,) The indices of the matched node pairs.
        /// \param  corr_scores_vec     (C,) The matching scores of each point correspondence.
        int match_instance_points(const std::string &ref_name,
                                const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                std::vector<Eigen::Vector3d> &corr_src_points,
                                std::vector<Eigen::Vector3d> &corr_ref_points,
                                std::vector<int> &corr_match_indices,
                                std::vector<float> &corr_scores_vec,
                                std::string dir="");

        // void clear();

        torch::Tensor get_active_node_feats()const{
            return src_features.node_features.clone();
        }

        /// \brief  Get the active node features in 2-level vectors.
        /// \param  feat_vector (N, D) The active node features.
        /// \param  N           The number of nodes.
        /// \param  D           The dimension of the node features.
        bool get_active_node_feats(std::vector<std::vector<float>> &feat_vector,
                                int &N, int &D)const;

        torch::Tensor get_ref_node_feats(const std::string &ref_name)const{
            if(ref_graphs.find(ref_name)==ref_graphs.end()){
                return torch::zeros({0,0},torch::kFloat32);
            }
            else{
                return ref_graphs.at(ref_name).node_features.clone();
            }
        }

        int get_ref_feats_number(const std::string &ref_name)const{
            if(ref_graphs.find(ref_name)==ref_graphs.end()){
                return 0;
            }
            else{
                return ref_graphs.at(ref_name).node_features.size(0);
            }
        }

        bool IsSrcShapeEmbedded()const{
            return src_features.shape_embedded;
        }

        bool IsRefShapeEmbedded(const std::string &ref_name)const{
            if(ref_graphs.find(ref_name)==ref_graphs.end()){
                return false;
            }
            else{
                return ref_graphs.at(ref_name).shape_embedded;
            }
        }

        bool save_middle_features(const std::string &dir);

    private:
        bool encode_scene_graph(const std::vector<NodePtr> &nodes, ImplicitGraph &graph_features);

        /// @brief  Reserve the features memories on GPU. 
        void initialize_graph_features();

    private:
        std::string cuda_device_string;
        ImplicitGraph src_features;    
        std::unordered_map<std::string, ImplicitGraph> ref_graphs;
        std::unordered_map<std::string, float> ref_sg_timestamps; // The latest received sg frame id

        LoopDetectorConfig config;
        std::shared_ptr<ShapeEncoder> shape_encoder;
        std::shared_ptr<SgNet> sgnet;
        std::string weight_folder_dir;

    };
    
} // namespace fmfusion
#endif