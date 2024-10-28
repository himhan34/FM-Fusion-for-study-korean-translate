#include "Graph.h"


namespace fmfusion
{
    void Node::sample_corners(const int &max_corner_number, std::vector<Corner> &corner_vector, int padding_value)
    {
        // std::vector<Corner> corner_vector;
        std::stringstream msg;
        msg<<corners.size()<<" corners: ";
        corner_vector.reserve(max_corner_number);
        Corner padding_corner = {(uint32_t)padding_value, (uint32_t)padding_value};

        if (corners.size()<max_corner_number){
            corner_vector.insert(corner_vector.end(), corners.begin(), corners.end());

            // zero padding
            for (int i=corners.size(); i<max_corner_number; i++){
                corner_vector.emplace_back(padding_corner);
            }

        }
        else if (corners.size()==max_corner_number){
            corner_vector.insert(corner_vector.end(), corners.begin(), corners.end());
        }
        else{
            std::vector<int> indices(corners.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());
            for (int i=0; i<max_corner_number; i++){
                corner_vector.emplace_back(corners[indices[i]]);
                // corner_array[i] = corners[indices[i]];
                // msg<< indices[i]<<",";
            }
        }
        assert(corner_vector.size()==max_corner_number);
        msg<<"\n";
        // std::cout<<msg.str();
    }

    Graph::Graph(GraphConfig config_):config(config_),max_corner_number(0),max_neighbor_number(0),frame_id(-1),timestamp(-1.0)
    {
        std::cout<<"GNN initialized.\n";

    }

    void Graph::initialize(const std::vector<InstancePtr> &instances)
    {
        o3d_utility::Timer timer_;
        timer_.Start();
        frame_id = 1;
        // std::cout<<"Constructing GNN...\n";
        for (auto inst:instances){
            NodePtr node = std::make_shared<Node>(nodes.size(), inst->get_id());
            std::string label = inst->get_predicted_class().first;
            if (config.ignore_labels.find(label)!=std::string::npos) continue;
            node->semantic = label;
            node->centroid = inst->centroid;
            node->bbox_shape = inst->min_box->extent_;
            node->cloud = std::make_shared<open3d::geometry::PointCloud>(*inst->point_cloud); // deep copy
            if (config.voxel_size>0.0)
                node->cloud = node->cloud->VoxelDownSample(config.voxel_size);
            
            nodes.push_back(node);
            node_instance_idxs.push_back(inst->get_id());
            instance2node_idx[inst->get_id()] = node->id;
        }
        timer_.Stop();
        std::cout<<"Constructed "<<nodes.size()<<" nodes in "
                        <<std::fixed<<std::setprecision(3)
                        <<timer_.GetDurationInMillisecond()<< " ms.\n";

    }

    int Graph::subscribe_coarse_nodes(const float &latest_timestamp,
                                        const std::vector<uint32_t> &node_indices,
                                        const std::vector<uint32_t> &instances,
                                        const std::vector<Eigen::Vector3d> &centroids)
    {
        if(latest_timestamp - timestamp > 0.01){
            clear();
            std::stringstream msg;
            for (int i=0; i<instances.size(); i++){
                NodePtr node = std::make_shared<Node>(node_indices[i], instances[i]);
                node->centroid = centroids[i];
                node->cloud = std::make_shared<open3d::geometry::PointCloud>();
                nodes.push_back(node);
                node_instance_idxs.push_back(instances[i]);
                instance2node_idx[instances[i]] = i;
                msg<<instances[i]<<",";
            }
            timestamp = latest_timestamp;

            return instances.size();          
        }
        else return 0;
    }

    int Graph::subscribde_dense_points(const float &sub_timestamp,
                                        const std::vector<Eigen::Vector3d> &xyz,
                                        const std::vector<uint32_t> &labels)
    {
        int count = 0; // count the points been updated
        int N = nodes.size();
        int X = xyz.size();
        std::vector<std::vector<Eigen::Vector3d>> nodes_points(N, std::vector<Eigen::Vector3d>());
        std::stringstream msg;
        for(int k=0;k<X;k++){
            if (labels[k]>=N){
                std::cout<<"Node "<<labels[k]<<" not found in the graph.\n";
                continue;
            }
            nodes_points[labels[k]].push_back(xyz[k]);
        }

        for(int i=0;i<N;i++){
            if (nodes_points[i].empty()) continue;
            nodes[i]->cloud = std::make_shared<open3d::geometry::PointCloud>(nodes_points[i]);
            count += nodes_points[i].size();
        }

        return count;
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
        if (config.involve_floor_edge && !floors.empty()){

            for (int i=0; i<N; i++){ // each instance connect to one closest floor
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
        // std::cout<<"Constructed "<<edges.size()<<" edges in "
        //     <<std::fixed<<std::setprecision(3)
        //     <<timer_.GetDurationInMillisecond()<<" ms.\n";

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
        // std::cout<<"Construct corners for all the nodes.\n"
        //     <<"The max nieghbor number is "<<max_neighbor_number<<"; "
        //     <<"The max corner number is "<<max_corner_number<<".\n";
        assert(max_corner_number>0);

    }

    void Graph::extract_global_cloud(std::vector<Eigen::Vector3d> &xyz,std::vector<uint32_t> &labels)
    {
        // std::vector<Eigen::Vector3d> xyz;
        // std::vector<int> labels;
        std::stringstream msg;
        msg<<"Nodes id: ";
        for(auto node:nodes){
            xyz.insert(xyz.end(), node->cloud->points_.begin(), node->cloud->points_.end());
            labels.insert(labels.end(), node->cloud->points_.size(), node->id);
            msg<<node->id<<",";
        }
        msg<<"\n";
        // std::cout<<msg.str();

    }

    O3d_Cloud_Ptr Graph::extract_global_cloud(float vx_size)const 
    {
        O3d_Cloud_Ptr out_cloud_ptr = std::make_shared<open3d::geometry::PointCloud>();
        for(auto node:nodes){
            *out_cloud_ptr += *(node->cloud);
        }
        if(vx_size>0.0) out_cloud_ptr->VoxelDownSample(vx_size);

        return out_cloud_ptr;
    }

    void Graph::clear()
    {
        nodes.clear();
        edges.clear();
        node_instance_idxs.clear();
        instance2node_idx.clear();
        max_corner_number = 0;
        max_neighbor_number = 0;
    }

    DataDict Graph::extract_data_dict(bool coarse)
    {
        DataDict data_dict;
        for (auto node:nodes){
            data_dict.centroids.push_back(node->centroid);
            data_dict.nodes.push_back(node->id);
            data_dict.instances.push_back(node->instance_id);  
            if(!coarse && node->cloud.use_count()>0){
                data_dict.xyz.insert(data_dict.xyz.end(), node->cloud->points_.begin(), node->cloud->points_.end());
                data_dict.labels.insert(data_dict.labels.end(), node->cloud->points_.size(), node->id);
            }
        }
        data_dict.length_vec = std::vector<int>(1, data_dict.xyz.size());
        return data_dict;
    }

    void Graph::paint_all_floor(const Eigen::Vector3d &color)
    {
        std::string floor_names = "floor. carpet.";
        for (auto node:nodes){
            if (floor_names.find(node->semantic)!=std::string::npos){
                node->cloud->PaintUniformColor(color);
            }
        }
    }

    const std::vector<Eigen::Vector3d> Graph::get_centroids()const
    {
        std::vector<Eigen::Vector3d> centroids;
        for (auto node:nodes){
            centroids.push_back(node->centroid);
        }
        return centroids;
    }

    const std::vector<std::pair<int,int>> Graph::get_edges()const
    {
        std::vector<std::pair<int,int>> edge_pairs;
        for (auto edge:edges){
            edge_pairs.push_back(std::make_pair(edge->src_id, edge->ref_id));
        }
        return edge_pairs;
    }

}