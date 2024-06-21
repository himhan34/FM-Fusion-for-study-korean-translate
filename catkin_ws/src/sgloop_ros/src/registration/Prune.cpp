#include "registration/Prune.h"

namespace Registration
{

void pruneInsOutliers(const fmfusion::RegistrationConfig &config,
                      const std::vector<fmfusion::NodePtr> &src_nodes,
                      const std::vector<fmfusion::NodePtr> &ref_nodes,
                      const std::vector<fmfusion::NodePair> &match_pairs,
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
    // std::cout << msg.str() << std::endl;
}

    
}
