#ifndef FMFUSION_GRAPH_H
#define FMFUSION_GRAPH_H

#include <Common.h>
#include <mapping/Instance.h>

namespace fmfusion
{

typedef std::array<uint32_t,2> Corner;

class Node
{
    public:
        Node(uint32_t node_id_, InstanceId instance_id_):
            id(node_id_), instance_id(instance_id_){};
        
        /// @brief 
        /// @param max_corner_number 
        /// @param corner_vector 
        /// @param padding_value For the corners less than max_corner_number, padding with this value. 
        void sample_corners(const int &max_corner_number, std::vector<Corner> &corner_vector, int padding_value=0);

        ~Node(){};
    public:
        uint32_t id; // node id, reordered from 0
        InstanceId instance_id; // corresponds to original instance id
        std::string semantic;
        std::vector<uint32_t> neighbors;
        std::vector<Corner> corners; // corner1, corner2. The anchor node and corner nodes form a triplet.
        O3d_Cloud_Ptr cloud;
        Eigen::Vector3d centroid;
        Eigen::Vector3d bbox_shape; // x,y,z

};
typedef std::shared_ptr<Node> NodePtr;


class Edge
{
public:
    Edge(const uint32_t &id1, const uint32_t &id2): src_id(id1), ref_id(id2) {
        distance = 0.0;
    }
    ~Edge() {};

public:
    uint32_t src_id; // start node
    uint32_t ref_id; // end node
    double distance;
};
typedef std::shared_ptr<Edge> EdgePtr;


struct DataDict{
    std::vector<Eigen::Vector3d> xyz; // (X,3)
    std::vector<uint32_t> labels; // (X,)
    std::vector<Eigen::Vector3d> centroids; // (N,3)
    std::vector<uint32_t> nodes; // (N,)
};

class Graph
{
    /// \brief  Graph neural network(GNN) representations. Constructed from explicit scene graph.
    public:
        Graph(GraphConfig config_);
        void initialize(const std::vector<InstancePtr> &instances);

        void construct_edges();

        void construct_triplets();

        const std::vector<NodePtr> get_const_nodes() const { return nodes; }

        const std::vector<EdgePtr> get_const_edges() const { return edges; }

        const std::vector<Eigen::Vector3d> get_centroids()const;

        /// \brief  Extract global point cloud from the graph.
        /// @param xyz Global point cloud.
        /// @param labels Node ids.
        void extract_global_cloud(std::vector<Eigen::Vector3d> &xyz,std::vector<uint32_t> &labels);

        ///
        DataDict extract_data_dict();

        ~Graph() {};

    private:
        void update_neighbors();

    public:
        int max_neighbor_number;
        int max_corner_number;        

    private:
        GraphConfig config;
        std::vector<InstanceId> node_instance_idxs; // corresponds to original instance ids
        std::vector<NodePtr> nodes;
        std::vector<EdgePtr> edges; // Each edge is a pair of node id


};

}


#endif

