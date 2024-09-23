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
    std::vector<int> length_vec; // (B,) [|X1|,|X2|,...,|XB|]
    std::vector<uint32_t> labels; // (X,)
    std::vector<Eigen::Vector3d> centroids; // (N,3)
    std::vector<uint32_t> nodes; // (N,)
    std::vector<uint32_t> instances; // (N,)
    std::string print_instances(){
        std::stringstream msg;
        for (auto instance:instances){
            msg<<instance<<",";
        }
        return msg.str();
    }

    void clear(){
        xyz.clear();
        length_vec.clear();
        labels.clear();
        centroids.clear();
        nodes.clear();
        instances.clear();
    }

};

class Graph
{
    /// \brief  Explicit Scene Graph Representations.
    public:
        Graph(GraphConfig config_);
        void initialize(const std::vector<InstancePtr> &instances);

        /// \brief Update the node information received from communication server.
        ///        It cleared the graph and insert.
        int subscribe_coarse_nodes(const float &latest_timestamp,
                                    const std::vector<uint32_t> &node_indices,
                                    const std::vector<uint32_t> &instances,
                                    const std::vector<Eigen::Vector3d> &centroids);

        /// \brief Update the dense point cloud information received from communication server.
        ///        It should be updated after the coarse nodes are updated.
        int subscribde_dense_points(const float &sub_timestamp,
                                    const std::vector<Eigen::Vector3d> &xyz,
                                    const std::vector<uint32_t> &labels);

        void construct_edges();

        void construct_triplets();

        const std::vector<NodePtr> get_const_nodes() const { return nodes; }

        const std::vector<EdgePtr> get_const_edges() const { return edges; }

        const std::vector<std::pair<int,int>> get_edges() const;

        const std::vector<Eigen::Vector3d> get_centroids()const;

        void paint_all_floor(const Eigen::Vector3d &color);

        /// \brief  Extract global point cloud from the graph.
        /// @param xyz Global point cloud.
        /// @param labels Node ids.
        void extract_global_cloud(std::vector<Eigen::Vector3d> &xyz,std::vector<uint32_t> &labels);

        /// \brief  Extract data dictionary.
        /// @param coarse If true, extract only nodes, instances and centroids data.
        DataDict extract_data_dict(bool coarse=false);

        O3d_Cloud_Ptr extract_global_cloud(float vx_size=-1.0)const;

        /// \brief  Extract instances and centroids.
        DataDict extract_coarse_data_dict();

        std::string print_nodes_names()const{
            std::stringstream ss;
            for (auto node: nodes) ss<<node->instance_id<<",";
            return ss.str();
        }

        float get_timestamp()const{return timestamp;}

        void clear();

        ~Graph() {};

    private:
        void update_neighbors();

    public:
        int max_neighbor_number;
        int max_corner_number;        
        
    private:
        GraphConfig config;
        std::map<InstanceId, int> instance2node_idx; // instance id to node idx
        std::vector<InstanceId> node_instance_idxs; // corresponds to original instance ids
        std::vector<NodePtr> nodes;
        std::vector<EdgePtr> edges; // Each edge is a pair of node id

        int frame_id; // The latest received sg frame id
        float timestamp; // The latest received sg timestamp


};
typedef std::shared_ptr<Graph> GraphPtr;

}


#endif

