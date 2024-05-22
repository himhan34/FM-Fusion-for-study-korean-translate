#include "Tools.h"

namespace fmfusion::visualization
{
    std::shared_ptr<const open3d::geometry::LineSet> draw_edges(
        const std::vector<NodePtr> &nodes, const std::vector<EdgePtr> &edges)
    {
        std::vector<Eigen::Vector3d> points;
        std::vector<Eigen::Vector2i> lines;

        for (const auto &edge : edges)
        {
            assert (edge->src_id < nodes.size());
            auto node1 = nodes[edge->src_id];
            auto node2 = nodes[edge->ref_id];

            Eigen::Vector3d center1 = node1->centroid;
            Eigen::Vector3d center2 = node2->centroid;

            points.push_back(center1);
            points.push_back(center2);

            lines.push_back(Eigen::Vector2i(points.size() - 2, points.size() - 1));
        }
        std::shared_ptr<const open3d::geometry::LineSet> line_set = std::make_shared<open3d::geometry::LineSet>(points,lines);

        return line_set;
    }

    std::shared_ptr<const open3d::geometry::LineSet> draw_instance_correspondences(
        const std::vector<Eigen::Vector3d> &src_centroids, const std::vector<Eigen::Vector3d> &ref_centroids)
    {
        std::vector<Eigen::Vector3d> points;
        std::vector<Eigen::Vector2i> lines;

        for (size_t i = 0; i < src_centroids.size(); i++)
        {
            Eigen::Vector3d center1 = src_centroids[i];
            Eigen::Vector3d center2 = ref_centroids[i];

            points.push_back(center1);
            points.push_back(center2);

            lines.push_back(Eigen::Vector2i(points.size() - 2, points.size() - 1));
        }
        std::shared_ptr<const open3d::geometry::LineSet> line_set = std::make_shared<open3d::geometry::LineSet>(points,lines);
        return line_set;
    }


}

