#include <iostream>
#include <memory>
#include <vector>
#include "open3d/Open3D.h"

#include "Common.h"
#include "tools/Tools.h"
#include "tools/IO.h"
#include "tools/Eval.h"
#include "tools/Utility.h"
#include "tools/g3reg_api.h"
#include "mapping/SemanticMapping.h"

#include "sgloop/Graph.h"
#include "sgloop/LoopDetector.h"

typedef fmfusion::NodePair NodePair;

struct ExplicitInstances {
    std::vector<fmfusion::InstanceId> names;
    std::vector<fmfusion::InstancePtr> instances;
    // std::vector<Eigen::Vector3d> xyz;
    // std::vector<uint32_t> labels;
};

void pruneInsOutliers(const fmfusion::RegistrationConfig &config,
                      const std::vector<fmfusion::NodePtr> &src_nodes,
                      const std::vector<fmfusion::NodePtr> &ref_nodes,
                      const std::vector<NodePair> &match_pairs,
                      std::vector<bool> &pruned_true_masks) {

//    存储配准单位
    std::vector<clique_solver::GraphVertex::Ptr> v1, v2;
//    一致性图的节点有多种类型
    clique_solver::VertexInfo vertex_info;
//    不同的节点会有不同的一致性测量函数
    vertex_info.type = clique_solver::VertexType::POINT;
//    一致性判断的阈值，可以设置为多个，从小到大，会得到多个内点结果。
//  由于Instance中心点不确定性较大，我们这里只选取一个比较大的阈值，不做精确估计
//  注意，阈值越大，图越稠密，最大团求解时间也会上升
    vertex_info.noise_bound_vec = config.noise_bound_vec; //{1.0};
//    初始化配准节点
    for (const auto &src_node: src_nodes) {
        const Eigen::Vector3d &center = src_node->centroid;
        v1.push_back(clique_solver::create_vertex(center, vertex_info));
    }
    for (const auto &ref_node: ref_nodes) {
        const Eigen::Vector3d &center = ref_node->centroid;
        v2.push_back(clique_solver::create_vertex(center, vertex_info));
    }

//    匹配的数量
    uint64_t num_corr = match_pairs.size();
//    TRIM(Translation Rotation Invariant Measurements)的数量
    uint64_t num_tims = num_corr * (num_corr - 1) / 2;
//    初始化一致性图和最大团
    std::vector<clique_solver::Graph> graphs;
    std::vector<std::vector<int>> max_cliques;
    int num_graphs = vertex_info.noise_bound_vec.size();
    for (int i = 0; i < num_graphs; ++i) {
        clique_solver::Graph graph;
        graph.clear();
        graph.setType(false);
        graph.populateVertices(num_corr);
        graphs.push_back(graph);
        max_cliques.push_back(std::vector<int>());
    }

//    构建多个一致性图
#pragma omp parallel for default(none) shared(num_corr, num_tims, v1, v2, graphs, match_pairs, num_graphs)
    for (size_t k = 0; k < num_tims; ++k) {
        size_t i, j;
//        一共需要进行num_tims次边的计算，第k次对应第i，j个节点
        std::tie(i, j) = clique_solver::k2ij(k, num_corr);
//        由于之前的noise_bound_vec会有多个阈值，因此results也是多个
        const auto &results = (*v1[match_pairs[j].first] - *v1[match_pairs[i].first])->consistent(
                *(*v2[match_pairs[j].second] - *v2[match_pairs[i].second]));
        for (int level = 0; level < num_graphs; ++level) {
//            判断是否通过一致性检验
            if (results(level) > 0.0) {
#pragma omp critical
                {
                    graphs[level].addEdge(i, j);
                }
            }
        }
    }


//  求解最大团
    clique_solver::MaxCliqueSolver::Params clique_params;
    clique_params.solver_mode = clique_solver::MaxCliqueSolver::CLIQUE_SOLVER_MODE::PMC_EXACT;
    int prune_level = 0;
//    渐进式求解每个图的最大团
    for (int level = 0; level < num_graphs; ++level) {
        clique_solver::MaxCliqueSolver mac_solver(clique_params);
        max_cliques[level] = mac_solver.findMaxClique(graphs[level], prune_level);
        prune_level = max_cliques[level].size();
    }
//    最大团的元素表示第几个匹配关系是正确的。排序，为了好看。
    for (int level = 0; level < num_graphs; ++level) {
        auto &clique = max_cliques[level];
        std::sort(clique.begin(), clique.end());
    }

    std::stringstream msg;
    msg << "The inlier match pairs are: \n";
//    默认只有一个noise bound，我们只看第一个最大团的结果
    for (auto i: max_cliques[0]) {
        auto pair = match_pairs[i];
        pruned_true_masks[i] = true;
        const auto &src_node = src_nodes[pair.first];
        const auto &ref_node = ref_nodes[pair.second];
        msg << "(" << pair.first << "," << pair.second << ") "
            << "(" << src_node->semantic << "," << ref_node->semantic << ")\n";
    }
    std::cout << msg.str() << std::endl;
}

bool save_match_results(const std::vector<std::pair<uint32_t, uint32_t>> &match_pairs,
                        const std::vector<float> &match_scores,
                        const std::string &output_file_dir) {
    std::ofstream output_file(output_file_dir);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open file: " << output_file_dir << std::endl;
        return false;
    }

    output_file << "# src_id,ref_id,score" << std::endl;
    for (size_t i = 0; i < match_scores.size(); i++) {
        output_file << match_pairs[i].first << "," << match_pairs[i].second << ","
                    << std::fixed << std::setprecision(3) << match_scores[i] << std::endl;
    }
    output_file.close();

    return true;

}

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
    input.close();
    return bytes;
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    std::string config_file = utility::GetProgramOptionAsString(argc, argv, "--config");
    std::string ref_map_dir = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
    std::string src_map_dir = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
    std::string weights_folder = utility::GetProgramOptionAsString(argc, argv, "--weights_folder");
    std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
    bool dense_match = utility::ProgramOptionExists(argc, argv, "--dense_match");
    bool prune_instance = utility::ProgramOptionExists(argc, argv, "--prune_instance");
    int viz_mode = utility::GetProgramOptionAsInt(argc, argv, "--viz_mode", 0);
    int cuda_number = utility::GetProgramOptionAsInt(argc, argv, "--cuda_number", 0);
    int n_iter = utility::GetProgramOptionAsInt(argc, argv, "--n_iter", 1);
    std::string debug_str = utility::GetProgramOptionAsString(argc, argv, "--debug");
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    float ds_voxel_size = utility::GetProgramOptionAsDouble(argc, argv, "--ds_voxel_size", 0.05);
    std::string ref_name = "agentB";

    bool fused = true;
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel(verbose));

    // init
    auto sg_config = fmfusion::utility::create_scene_graph_config(config_file, true);
    if (sg_config == nullptr) {
        utility::LogWarning("Failed to create scene graph config.");
        return 0;
    }

    // Load Map
    auto ref_map = std::make_shared<fmfusion::SemanticMapping>(fmfusion::SemanticMapping(sg_config->mapping_cfg, sg_config->instance_cfg));
    auto src_map = std::make_shared<fmfusion::SemanticMapping>(fmfusion::SemanticMapping(sg_config->mapping_cfg, sg_config->instance_cfg));

    ref_map->load(ref_map_dir);
    src_map->load(src_map_dir);
    ref_map->extract_bounding_boxes();
    src_map->extract_bounding_boxes();

    // Export instances
    fmfusion::o3d_utility::Timer timer;
    ExplicitInstances ref_instances, src_instances;
    ref_map->export_instances(ref_instances.names, ref_instances.instances);
    src_map->export_instances(src_instances.names, src_instances.instances);
    assert(src_instances.names.size() > 0 && ref_instances.names.size() > 0);

    // Construct GNN from the exported instances
    auto ref_graph = std::make_shared<fmfusion::Graph>(sg_config->graph);
    auto src_graph = std::make_shared<fmfusion::Graph>(sg_config->graph);

    ref_graph->initialize(ref_instances.instances);
    src_graph->initialize(src_instances.instances);
    ref_graph->construct_edges();
    src_graph->construct_edges();
    timer.Start();
    ref_graph->construct_triplets();
    timer.Stop();
    src_graph->construct_triplets();
    std::cout << "construct triplets takes " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond()
              << " ms\n";
    fmfusion::DataDict ref_data_dict = ref_graph->extract_data_dict();
    fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();
    // std::cout<<"ref instance names: "<<ref_data_dict.print_instances()<<std::endl;

    if (debug_str.size() > 0) {
        torch::Device device(torch::kCUDA, cuda_number);
        std::cout << "Load script from " << weights_folder + "/" + debug_str << "\n";
        torch::jit::script::Module tmp_script = torch::jit::load(weights_folder + "/" + debug_str);
        tmp_script.to(device);

        // sleep 2 seconds
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return 0;
    }

    if (false) { // test load bert bag of words
        fmfusion::BertBow bert_bow(weights_folder + "/bert_bow.txt",
                                   weights_folder + "/bert_bow.pt");

        std::vector<char> f = get_the_bytes("/home/cliuci/tmp/test.pt");
        torch::IValue x = torch::pickle_load(f);
        torch::Tensor x_tensor = x.toTensor();
        std::cout << "Load tensor \n" << x_tensor << "\n";

        return 0;
    }

    // Encode using SgNet
    assert(cuda_number < torch::cuda::device_count());
    auto loop_detector = std::make_shared<fmfusion::LoopDetector>(fmfusion::LoopDetector(sg_config->loop_detector,
                                                                                         sg_config->shape_encoder,
                                                                                         sg_config->sgnet,
                                                                                         weights_folder,
                                                                                         cuda_number,
                                                                                         {ref_name}));

    torch::Tensor ref_nodes_feats, src_nodes_feats;
    torch::Tensor tmp_feats;

    for (int itr = 0; itr < n_iter; itr++) {
        utility::LogWarning("Iteration: {}", itr);
        timer.Start();
        loop_detector->encode_src_scene_graph(src_graph->get_const_nodes());
        timer.Stop();
        std::cout << "Encode src GNN " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond()
                  << " ms\n";

        timer.Start();
        loop_detector->encode_ref_scene_graph(ref_name, ref_graph->get_const_nodes());
        timer.Stop();
        std::cout << "Encode ref GNN " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond()
                  << " ms\n";

    }

    if (false) {   // test tensor->vector conversion
        torch::Tensor src_node_feats = loop_detector->get_active_node_feats().to(torch::kCPU);

        timer.Start();
        int Ns, Ds;
        std::vector<std::vector<float>> src_node_feats_vec;
        loop_detector->get_active_node_feats(src_node_feats_vec, Ns, Ds);
        timer.Stop();
        std::cout << "Export " << Ns << " x " << Ds
                  << " features takes " << timer.GetDurationInMillisecond() << " ms\n";

        // vector->array
        float src_node_feats_array[Ns][Ds];
        for (int i = 0; i < Ns; i++) {
            std::copy(src_node_feats_vec[i].begin(), src_node_feats_vec[i].end(), src_node_feats_array[i]);
        }

        // array->tensor
        torch::Tensor received_node_feats = torch::from_blob(src_node_feats_array,
                                                             {Ns, Ds}, torch::kFloat32);
        std::cout << "recreate node features using feature vectors\n";
        std::cout << "sent nan node: " << torch::isnan(src_node_feats).sum().item<int>() << "\n"
                  << "received nan node: " << torch::isnan(received_node_feats).sum().item<int>() << "\n";

        // Check the received node feature is close
        auto diff = received_node_feats - src_node_feats;
        std::cout << "Check the received node feature is close: "
                  << torch::allclose(received_node_feats, src_node_feats, 1e-6) << "\n";
        std::cout << "Sum difference " << torch::sum(diff).item<float>() << "\n";
        return 0;
    }

    // Shape encoder
    timer.Start();
    if (dense_match)
        loop_detector->encode_concat_sgs(ref_name, ref_graph->get_const_nodes().size(), ref_data_dict,
                                         src_graph->get_const_nodes().size(), src_data_dict, fused);
    timer.Stop();
    std::cout << "Encode stacked graph takes " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond()
              << " ms\n";

    // Hierachical matching
    std::vector<NodePair> match_pairs;
    std::vector<float> match_scores;
    std::vector<Eigen::Vector3d> corr_src_points, corr_ref_points; // (C,3),(C,3)
    std::vector<int> corr_match_indices; // (C,)
    std::vector<float> corr_scores_vec; // (C,)

    int M; // number of matched nodes
    int C = 0; // number of matched points
    timer.Start();
    M = loop_detector->match_nodes(ref_name, match_pairs, match_scores, fused);
    timer.Stop();
    std::cout << "Find " << M << " match. It takes " << std::fixed << std::setprecision(3)
              << timer.GetDurationInMillisecond() << " ms\n";

    std::vector<NodePair> pruned_match_pairs;
    std::vector<float> pruned_match_scores;
    if (prune_instance) {
        std::vector<bool> pruned_true_masks(M, false);
        timer.Start();
        pruneInsOutliers(sg_config->reg,
                         src_graph->get_const_nodes(),
                         ref_graph->get_const_nodes(),
                         match_pairs,
                         pruned_true_masks);
        timer.Stop();
        pruned_match_pairs = fmfusion::utility::update_masked_vec(match_pairs, pruned_true_masks);
        pruned_match_scores = fmfusion::utility::update_masked_vec(match_scores, pruned_true_masks);
    } else {
        pruned_match_pairs = match_pairs;
        pruned_match_scores = match_scores;
    }

    std::cout << "Prune nodes outliers takes "
              << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() << " ms\n";
    std::cout << "Keep " << pruned_match_pairs.size() << " consistent matched nodes\n";

    if (dense_match && M > 0) { // Dense match
        timer.Start();
        C = loop_detector->match_instance_points(ref_name,
                                                 pruned_match_pairs, 
                                                 corr_src_points,
                                                 corr_ref_points, 
                                                 corr_match_indices,
                                                 corr_scores_vec);
        timer.Stop();
        std::cout << "Match points takes " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond()
                  << " ms\n";
    }

    // Estimate pose
    std::vector<std::pair<fmfusion::InstanceId, fmfusion::InstanceId>> match_instances;
    std::vector<Eigen::Vector3d> src_centroids, ref_centroids;
    fmfusion::O3d_Cloud_Ptr src_cloud_ptr, ref_cloud_ptr;
    Eigen::Matrix4d pred_pose;
    fmfusion::IO::extract_match_instances(
            pruned_match_pairs, src_graph->get_const_nodes(), ref_graph->get_const_nodes(), match_instances);
    fmfusion::IO::extract_instance_correspondences(
            src_graph->get_const_nodes(), ref_graph->get_const_nodes(), pruned_match_pairs, pruned_match_scores,
            src_centroids,
            ref_centroids);
    src_cloud_ptr = src_map->export_global_pcd(true, ds_voxel_size);
    ref_cloud_ptr = ref_map->export_global_pcd(true, ds_voxel_size);

    g3reg::Config config;
//    noise bound的取值
    config.set_noise_bounds({0.2, 0.3});
//    位姿求解优化器的类型
    config.tf_solver = "quatro";
//    基于点到平面的距离做验证
    config.verify_mtd = "plane_based";
    G3RegAPI g3reg(config);

    std::cout << "G3Reg "
              << pruned_match_pairs.size() << " nodes, "
              << corr_src_points.size() << " points\n";
    timer.Start();
    g3reg.estimate_pose(src_graph->get_const_nodes(),
                        ref_graph->get_const_nodes(),
                        pruned_match_pairs,
                        corr_scores_vec,
                        corr_src_points,
                        corr_ref_points,
                        src_cloud_ptr,
                        ref_cloud_ptr,
                        pred_pose);
    timer.Stop();
    double g3reg_time = timer.GetDurationInMillisecond();
    timer.Start();
    std::cout << "Continue to use ICP to refine the pose\n";
    pred_pose = g3reg.icp_refine(src_cloud_ptr, ref_cloud_ptr, pred_pose);
    timer.Stop();
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() + g3reg_time
              << " ms, G3Reg time: " << std::fixed << std::setprecision(3) << g3reg_time << " ms, ICP time: "
              << std::fixed << std::setprecision(3) << timer.GetDurationInMillisecond() << " ms\n";

    // Eval
    std::vector<bool> matches_true_masks;
    // if (gt_file.size() > 0) {
    //     int true_instance_match = fmfusion::maks_true_instance(gt_file, match_instances, matches_true_masks);
    //     std::cout << "True instance match: " << true_instance_match << "/" << match_instances.size() << std::endl;
    // }

    // visualization
    if (viz_mode == 1) { // instance match
        auto src_geometries = src_map->get_geometries(true, false);
        auto ref_geometries = ref_map->get_geometries(true, false);
        auto instance_match_lineset = fmfusion::visualization::draw_instance_correspondences(src_centroids,
                                                                                             ref_centroids);

        std::vector<fmfusion::O3d_Geometry_Ptr> viz_geometries;
        viz_geometries.insert(viz_geometries.end(), src_geometries.begin(), src_geometries.end());
        viz_geometries.insert(viz_geometries.end(), ref_geometries.begin(), ref_geometries.end());
        viz_geometries.emplace_back(instance_match_lineset);
        open3d::visualization::DrawGeometries(viz_geometries, "UST_RI", 1920, 1080);
    } else if (viz_mode == 2) {   // registration result
        src_cloud_ptr->Transform(pred_pose);
        src_cloud_ptr->PaintUniformColor(Eigen::Vector3d(0.0, 0.651, 0.929));
        ref_cloud_ptr->PaintUniformColor(Eigen::Vector3d(1.0, 0.706, 0));
        std::vector<fmfusion::O3d_Geometry_Ptr> viz_geometries = {src_cloud_ptr, ref_cloud_ptr};
        open3d::visualization::DrawGeometries(viz_geometries, "UST_RI", 1920, 1080);
    }

    if (output_folder.size() > 0) {
        // save the instance correspondences
        std::string ref_scene = ref_map_dir.substr(ref_map_dir.find_last_of("/") + 1);
        std::string src_scene = src_map_dir.substr(src_map_dir.find_last_of("/") + 1);
        std::string pair_name = src_scene + "-" + ref_scene;
        std::string output_file_dir = output_folder + "/" + pair_name + ".txt";
        fmfusion::IO::save_match_results(pred_pose, match_instances, pruned_match_scores, output_file_dir);
        open3d::io::WritePointCloudToPLY(output_folder + "/" + src_scene + ".ply", *src_cloud_ptr, {});

        if (C > 0) {
            fmfusion::O3d_Cloud_Ptr corr_src_pcd = std::make_shared<fmfusion::O3d_Cloud>(corr_src_points);
            fmfusion::O3d_Cloud_Ptr corr_ref_pcd = std::make_shared<fmfusion::O3d_Cloud>(corr_ref_points);
            open3d::io::WritePointCloudToPLY(output_folder + "/" + pair_name + "_csrc.ply", *corr_src_pcd, {});
            open3d::io::WritePointCloudToPLY(output_folder + "/" + pair_name + "_cref.ply", *corr_ref_pcd, {});
            fmfusion::IO::save_corrs_match_indices(corr_match_indices, 
                                                    output_folder + "/" + pair_name + "_cmatches.txt");
        }

        std::cout << "Save output result to " << output_folder << std::endl;
    }

    fmfusion::utility::write_config(output_folder + "/config.txt", *sg_config);

    return 0;
}

