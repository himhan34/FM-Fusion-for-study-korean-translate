#ifndef FMFUSION_TOOLS_H
#define FMFUSION_TOOLS_H

#include "mapping/Instance.h"
#include "sgloop/Graph.h"

namespace fmfusion
{

namespace visualization
{
/// \brief  Visualization functions in Open3D lib.

    /// \brief  Draw intra-graph edges between instances.
    std::shared_ptr<const open3d::geometry::LineSet> draw_edges(
        const std::vector<NodePtr> &nodes, const std::vector<EdgePtr> &edges);
    

    std::shared_ptr<const open3d::geometry::LineSet> draw_instance_correspondences(
        const std::vector<Eigen::Vector3d> &src_centroids, const std::vector<Eigen::Vector3d> &ref_centroids);

}


}


#endif // FMFUSION_TOOLS_H



