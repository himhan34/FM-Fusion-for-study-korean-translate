#include "Graph.h"


namespace fmfusion
{
    Graph::Graph(GraphConfig config_):config(config_),max_corner_number(0),max_neighbor_number(0)
    {
        std::cout<<"GNN initialized.\n";

    }

    void Graph::initialize(const std::vector<InstancePtr> &instances)
    {
        o3d_utility::Timer timer_;
        timer_.Start();
        std::cout<<"Constructing GNN...\n";
        for (auto inst:instances){
            NodePtr node = std::make_shared<Node>(nodes.size(), inst->get_id());
            node->semantic = inst->get_predicted_class().first;
            node->centroid = inst->centroid;
            node->bbox_shape = inst->min_box->extent_;
            node->cloud = std::make_shared<open3d::geometry::PointCloud>(*inst->point_cloud); // deep copy


            nodes.push_back(node);
            node_instance_idxs.push_back(inst->get_id());
        }
        timer_.Stop();
        std::cout<<"Constructed "<<nodes.size()<<" nodes in "<<timer_.GetDurationInMillisecond()<< " ms.\n";

    }

    void Graph::construct_edges()
    {
        std::string floor_names = "floor. carpet.";
        std::string ceiling_names = "ceiling.";
        std::set<uint32_t> floors, ceilings; // store index
        const int N = nodes.size();
        std::stringstream msg;
        std::stringstream edge_msgs;
        float MIN_SEARCH_RADIUS = 1.0;
        float MAX_SEARCH_RADIUS = 6.0;
        o3d_utility::Timer timer_;
        timer_.Start();

        // Object to Object
        for (int i=0; i<N; i++){
            const NodePtr src = nodes[i];
            float radius_src;
            if (floor_names.find(src->semantic) != std::string::npos){
                floors.emplace(i);
                continue;
            }
            if (ceiling_names.find(src->semantic) != std::string::npos){
                ceilings.emplace(i);
                continue;
            }

            if (src->semantic.find("wall")!=std::string::npos)
                radius_src = MIN_SEARCH_RADIUS;
            else
                radius_src = src->bbox_shape.norm()/2.0;
            msg<<src->semantic <<"("<<radius_src<<"): ";
            for (int j=i+1; j<N; j++){
                const NodePtr ref = nodes[j];
                // std::string ref_label = ref->get_predicted_class().first;
                if(src->id == ref->id || 
                    floor_names.find(ref->semantic) != std::string::npos||
                    ceiling_names.find(ref->semantic) != std::string::npos)
                    continue;
                msg<<ref->semantic<<"";
                float radius_ref;
                if (ref->semantic.find("wall")!=std::string::npos)
                    radius_ref = MIN_SEARCH_RADIUS;
                else
                    radius_ref = ref->bbox_shape.norm()/2.0;
                float search_radius = config.edge_radius_ratio * std::max(radius_src, radius_ref);
                search_radius = std::max(std::min(search_radius, MAX_SEARCH_RADIUS),MIN_SEARCH_RADIUS);

                float dist = (src->centroid - ref->centroid).norm();
                msg<<"("<<radius_ref<<")("<<search_radius<<"),";
                if(dist<search_radius){
                    EdgePtr edge = std::make_shared<Edge>(i,j); //(src->id_, ref->id_);
                    edges.push_back(edge);
                    edge_msgs<<"("<<src->instance_id<<","<<ref->instance_id<<"),";
                }
            }
            msg<<"\n";
        }
        edge_msgs<<"\n";
        // std::cout<<edge_msgs.str();

        //
        if (config.involve_floor && !floors.empty()){

            for (int i=0; i<N; i++){
                if (floors.find(i)!=floors.end()) continue;
                std::pair<int,float> closet_floor = std::make_pair(-1,1000000.0);
                const NodePtr src = nodes[i];
                for (auto floor_index:floors){
                    float dist = (src->centroid - nodes[floor_index]->centroid).norm();
                    if (dist<closet_floor.second)
                        closet_floor = std::make_pair(floor_index,dist);
                }

                //
                if(closet_floor.first>=0){ // valid edge
                    EdgePtr edge = std::make_shared<Edge>(i, closet_floor.first);
                    edges.push_back(edge);
                }
            }


        }

        timer_.Stop();
        std::cout<<"Constructed "<<edges.size()<<" edges in "<<timer_.GetDurationInMillisecond()<<" ms.\n";

        update_neighbors();
    }

    void Graph::update_neighbors()
    {
        for (auto edge:edges){
            nodes[edge->src_id]->neighbors.push_back(edge->ref_id);
            nodes[edge->ref_id]->neighbors.push_back(edge->src_id);
        }
    }

    void Graph::construct_triplets()
    {
        for (NodePtr &node:nodes){
            // std::cout<<node->neighbors.size()<<" neighbors"<<std::endl;
            std::stringstream neighbor_msg, corner_msg;
            if(node->neighbors.size()<2) continue;
            for (int i=0; i<node->neighbors.size(); i++){
                neighbor_msg<<node->neighbors[i]<<",";
                for (int j=i+1; j<node->neighbors.size(); j++){
                    Corner corner = {node->neighbors[i], node->neighbors[j]};
                    node->corners.push_back(corner);
                    corner_msg<<"("<<corner[0]<<","<<corner[1]<<"),";
                }
            }
            // std::cout<<node->corners.size()<<" corners"<<std::endl;
            if (node->neighbors.size()>max_neighbor_number) max_neighbor_number = node->neighbors.size();
            if (node->corners.size()>max_corner_number) max_corner_number = node->corners.size();
            // std::cout<<"  "<<node->id<<": "<<neighbor_msg.str()<<std::endl;
            // std::cout<<"  "<<node->corners.size()<<" corners: "<<corner_msg.str()<<std::endl;
        }
        std::cout<<"Construct corners for all the nodes.\n"
            <<"The max nieghbor number is "<<max_neighbor_number<<"; "
            <<"The max corner number is "<<max_corner_number<<".\n";
    }


}