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
        LoopDetector(ShapeEncoderConfig &shape_encoder_config, SgNetConfig &sgnet_config, const std::string weight_folder);
        ~LoopDetector() {};

        bool encode_ref_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict);

        bool encode_src_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict);

        int match_nodes(std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            std::vector<float> &match_scores);

        /// \brief  Match the points of the corresponding instances.
        /// \param  match_pairs         The matched node pairs.
        /// \param  corr_src_points     (C,3) The corresponding points in the source instance.
        /// \param  corr_ref_points     (C,3) The corresponding points in the reference instance.
        /// \param  corr_scores_vec     (C,) The matching scores of each point correspondence.
        int match_instance_points(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                                std::vector<Eigen::Vector3d> &corr_src_points,
                                std::vector<Eigen::Vector3d> &corr_ref_points,
                                std::vector<float> &corr_scores_vec);

    private:
        bool encode_scene_graph(const std::vector<NodePtr> &nodes,const DataDict &data_dict, ImplicitGraph &graph_features);

    private:
        ImplicitGraph ref_features;
        ImplicitGraph src_features;    

        LoopDetectorConfig config;
        std::shared_ptr<ShapeEncoder> shape_encoder;
        std::shared_ptr<SgNet> sgnet;


    };
    
} // namespace fmfusion
#endif