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
                    const std::string weight_folder);
        ~LoopDetector() {};

        /// \brief  Encode the reference scene graph. 
        /// If \param data_dict is empty, it only run triplet gnn and skip dense point cloud encoding.
        bool encode_ref_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict);

        /// \brief  Encode the source scene graph. 
        /// If \param data_dict is empty, it only run triplet gnn and skip dense point cloud encoding.
        bool encode_src_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict);

        /// \brief Update the ref sg features from subscribed com server.
        bool subscribe_ref_coarse_features(const float &latest_timestamp, 
                                        const std::vector<std::vector<float>> &coarse_features_vec,
                                        torch::Tensor coarse_features);

        /// Concatenate the point cloud and instance points from two scene graphs to encode.
        bool encode_concat_sgs(const int &Nr, const DataDict& ref_data_dict,
                                const int& Ns, const DataDict& src_data_dict,
                                bool fused=false);

        int match_nodes(std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            std::vector<float> &match_scores,bool fused=false);

        /// \brief  Match the points of the corresponding instances.
        /// \param  match_pairs         The matched node pairs.
        /// \param  corr_src_points     (C,3) The corresponding points in the source instance.
        /// \param  corr_ref_points     (C,3) The corresponding points in the reference instance.
        /// \param  corr_scores_vec     (C,) The matching scores of each point correspondence.
        int match_instance_points(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                std::vector<Eigen::Vector3d> &corr_src_points,
                                std::vector<Eigen::Vector3d> &corr_ref_points,
                                std::vector<float> &corr_scores_vec);

        void clear();

        torch::Tensor get_active_node_feats()const{
            return src_features.node_features.clone();
        }

        /// \brief  Get the active node features in 2-level vectors.
        /// \param  feat_vector (N, D) The active node features.
        /// \param  N           The number of nodes.
        /// \param  D           The dimension of the node features.
        bool get_active_node_feats(std::vector<std::vector<float>> &feat_vector,
                                int &N, int &D)const;

        torch::Tensor get_ref_node_feats()const{
            return ref_features.node_features.clone();
        }

        int get_ref_feats_number()const{
            return ref_features.node_features.size(0);
        }

    private:
        bool encode_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict, ImplicitGraph &graph_features);

    private:
        ImplicitGraph ref_features;
        ImplicitGraph src_features;    

        LoopDetectorConfig config;
        std::shared_ptr<ShapeEncoder> shape_encoder;
        std::shared_ptr<SgNet> sgnet;
        std::string weight_folder_dir;

        float ref_sg_timestamp; // The latest received sg frame id

    };
    
} // namespace fmfusion
#endif