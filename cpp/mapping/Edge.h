#ifndef FMFUSION_EDGE_H 
#define FMFUSION_EDGE_H

#include <set>
#include "Instance.h"


namespace fmfusion
{
    class Edge
    {
    public:
        Edge(const InstanceId &id1, const InstanceId &id2): src_id(id1), ref_id(id2) {
            distance = 0.0;
        }
        ~Edge() {};

    public:
        InstanceId get_src_id() const { return src_id; }
        InstanceId get_ref_id() const { return ref_id; }
        double get_dist() const { return distance; }

    private:
        InstanceId src_id; // start node
        InstanceId ref_id; // end node
        double distance;
    };

    typedef std::shared_ptr<Edge> EdgePtr;

    /// \brief  Construct directed edges between instances. The reverse edges are not included.
    ///         Search {object,object} and {object,floor} edges.
    bool construct_edges(const std::vector<InstancePtr> &instances,
                                std::vector<EdgePtr> &edges, float radius_ratio=2.0, bool involve_floor=false);

}




#endif

