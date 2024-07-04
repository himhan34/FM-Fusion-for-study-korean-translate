#include "communication/Communication.h"

namespace SgCom
{
    void convert_tensor(const std::vector<std::vector<float>> &features_vector, 
                                            const int &N, const int &D, torch::Tensor &features_tensor)
    {
        // std::cout<<"Construct tensor array in shape "<<N<<" x "<<D<<"\n";
        float features_array[N][D];

        for(int i=0; i<N; i++){
            std::copy(features_vector[i].begin(), features_vector[i].end(), features_array[i]);
        }

        features_tensor = torch::ones({N, D}, torch::kFloat32); //test
        // features_tensor = torch::from_blob(features_array, {N, D}, torch::kFloat32);
        std::cout<<"nan number in convert: "<<torch::isnan(features_tensor).sum().item<int>()<<"\n";
    }   

    Communication::Communication(ros::NodeHandle &nh, ros::NodeHandle &nh_private,
                                std::string master_agent, 
                                std::vector<std::string> other_agents):
                                    local_agent(master_agent), pub_dense_msg(false), pub_dense_frame_id(-1)
    {
        typedef const boost::function<void(const sgloop_ros::DenseGraphConstPtr &)>  dense_callback_fn;
        typedef const boost::function<void(const std_msgs::Int32ConstPtr &)>  request_dense_callback_fn;

        pub_dense_sg = nh.advertise<sgloop_ros::DenseGraph>(local_agent + "/scene_graph", 1000);
        request_dense_callback_fn bounded_request_dense_callback = boost::bind(&Communication::request_dense_callback, this, _1);
        sub_request_dense = nh.subscribe(local_agent + "/request_dense", 1000, bounded_request_dense_callback);
        // sub_request_dense = nh.subscribe(local_agent + "/request_dense", 1000, Communication::request_dense_callback);

        for(const std::string &agent: other_agents){
            dense_callback_fn bounded_dense_callback = boost::bind(&Communication::dense_graph_callback, this, _1, agent);
            ros::Subscriber dense_listener = nh.subscribe(agent + "/scene_graph", 1000, bounded_dense_callback);
            ros::Publisher req_dense_pub = nh.advertise<std_msgs::Int32>(agent + "/request_dense", 1000);
            agents_dense_subscriber.push_back(dense_listener);
            pub_agents_request_dense[agent] = req_dense_pub;
            agents_data_dict[agent] = AgentDataDict{};
        }
        ROS_WARN("Init com for master agent: %s\n",local_agent.c_str());
        // std::cout<<"Init com for master agent: "<<local_agent<<"\n";

        last_exchange_frame_id = -1;
        frame_duration = 0.1;
        time_offset = 12000;
    }

    /*    
    bool Communication::broadcast_coarse_graph(int frame_id,
                                const std::vector<uint32_t> &instances,
                                const std::vector<Eigen::Vector3d> &centroids,
                                const int& N, const int &D,
                                const std::vector<std::vector<float>> &features)
    {
        sgloop_ros::CoarseGraph graph_msg;
        // std_msgs::Float32MultiArray feat_msg;
        // std::cout<<instances.size()<<" "<<centroids.size()<<" "<<N<<" "<<D<<" "<<features.size()<<"\n";
        assert(instances.size() == centroids.size());
        assert(instances.size() == N);
        //
        graph_msg.header.seq = frame_id;
        double timestamp = time_offset + frame_id * frame_duration;
        graph_msg.header.stamp = ros::Time(timestamp);
        graph_msg.header.frame_id = local_agent;

        // write instances into coarse graph msg
        graph_msg.instances.reserve(instances.size());
        graph_msg.centroids.reserve(instances.size());
        for(int i=0;i<N;i++){
            graph_msg.instances.push_back(instances[i]);
            geometry_msgs::Point pt;
            pt.x = centroids[i](0);
            pt.y = centroids[i](1);
            pt.z = centroids[i](2);
            graph_msg.centroids.push_back(pt);
        }

        // write features into feat msg
        graph_msg.features.layout.dim.resize(2, std_msgs::MultiArrayDimension());
        graph_msg.features.layout.dim[0].label = 'N';
        graph_msg.features.layout.dim[0].size = N;
        graph_msg.features.layout.dim[0].stride = N * D;
        graph_msg.features.layout.dim[1].label = 'D';
        graph_msg.features.layout.dim[1].size = D;
        graph_msg.features.layout.dim[1].stride = D;
        graph_msg.features.layout.data_offset = 0;

        graph_msg.features.data.reserve(N*D);
        for(const auto &feat_vec: features){
            for(const auto &ele: feat_vec){
                graph_msg.features.data.push_back(ele);
            }
        }

        //
        pub_coarse_sg.publish(graph_msg);

        //
        Log log;
        log.direction="pub";
        log.msg_type="CoarseGraph";
        log.frame_id = frame_id;
        log.timestamp = timestamp;
        log.checksum = true;
        pub_logs.push_back(log);

        return true;
    }
    */

    bool Communication::broadcast_dense_graph(int frame_id,
                                const std::vector<uint32_t> &nodes,
                                const std::vector<uint32_t> &instances,
                                const std::vector<Eigen::Vector3d> &centroids,
                                const std::vector<std::vector<float>> &features,
                                const std::vector<Eigen::Vector3d> &xyz,
                                const std::vector<uint32_t> &labels)
    {
        sgloop_ros::DenseGraph graph_msg;
        int N = instances.size();
        int X = xyz.size();
        int D = features[0].size(); // coarse feature dimension
        assert(N == centroids.size() && N == nodes.size() && N == features.size());
        assert(X == labels.size());
        if(N<1) {
            ROS_WARN("Error: empty nodes or points in publishing dense graph. Drop it.\n");
            return false;
        }

        graph_msg.header.seq = frame_id;
        double timestamp = time_offset + frame_id * frame_duration;
        graph_msg.header.stamp = ros::Time(timestamp);
        graph_msg.header.frame_id = local_agent;
        graph_msg.nodes_number = N;
        graph_msg.points_number = X;

        // write nodes info
        graph_msg.nodes.reserve(N);
        graph_msg.instances.reserve(N);
        graph_msg.centroids.reserve(N);
        std::stringstream msg;
        for(int i=0;i<N;i++){
            graph_msg.nodes.push_back(nodes[i]);
            graph_msg.instances.push_back(instances[i]);
            geometry_msgs::Point pt;
            pt.x = centroids[i](0);
            pt.y = centroids[i](1);
            pt.z = centroids[i](2);
            graph_msg.centroids.push_back(pt);
            msg<<"node: "<<nodes[i]<<" instance: "<<instances[i]<<"\n";
        }
        // std::cout<<msg.str();

        // write nodes features
        graph_msg.features.layout.dim.resize(2, std_msgs::MultiArrayDimension());
        graph_msg.features.layout.dim[0].label = 'N';
        graph_msg.features.layout.dim[0].size = N;
        graph_msg.features.layout.dim[0].stride = N * D;
        graph_msg.features.layout.dim[1].label = 'D';
        graph_msg.features.layout.dim[1].size = D;
        graph_msg.features.layout.dim[1].stride = D;
        graph_msg.features.layout.data_offset = 0;
        graph_msg.features.data.reserve(N*D);

        for(const auto &feat_vec: features){
            for(const auto &ele: feat_vec){
                graph_msg.features.data.push_back(ele);
            }
        }

        // write dense xyz and labels
        if(X>0){
            graph_msg.points.reserve(X);
            graph_msg.labels.reserve(X);
            for(int k=0;k<X;k++){
                geometry_msgs::Point pt;
                pt.x = xyz[k](0);
                pt.y = xyz[k](1);
                pt.z = xyz[k](2);
                graph_msg.points.push_back(pt);
                graph_msg.labels.push_back(labels[k]);
            }
        }
        //
        pub_dense_sg.publish(graph_msg);
        // if(X>0)
        //     ROS_WARN("Publish dense graph msg %d nodes and %d point\n",N,X);
        // else
        //     ROS_WARN("Publish coarse graph msg %d node\n",N);

        //
        Log log;
        log.direction="pub";
        if(X>0) log.msg_type="DenseGraph";
        else log.msg_type="CoarseGraph";
        log.frame_id = frame_id;
        log.timestamp = timestamp;
        log.checksum = true;
        log.N = N;
        log.X = X;
        pub_logs.push_back(log);

        return true;
    }

    /*
    void Communication::coarse_graph_callback(const sgloop_ros::CoarseGraph::ConstPtr &msg,
                                            std::string agent_name)
    {
        ROS_WARN("Received coarse graph msg from agent %s\n",agent_name.c_str());
        // ROS_WARN("Received coarse graph msg\n");

        int recv_frame_id = msg->header.seq;
        float rece_timestamp = msg->header.stamp.toSec();
        Log log;
        log.direction="sub";
        log.msg_type="CoarseGraph";
        log.frame_id = msg->header.seq;
        log.timestamp = rece_timestamp;

        int N = msg->features.layout.dim[0].size;
        int D = msg->features.layout.dim[1].size;
        if(msg->instances.size() != msg->centroids.size() 
            || msg->instances.size()!=N 
            ||msg->features.data.size()!=N*D){
            ROS_WARN("Error: inconsistent instance and centroid size. Drop it.\n");
            log.checksum = false;
            sub_logs.push_back(log);
            return;
        }

        // Read
        assert(agents_data_dict.find(agent_name) != agents_data_dict.end());
        AgentDataDict &data_dict = agents_data_dict[agent_name];
        data_dict.clear();
        data_dict.frame_id = recv_frame_id;
        data_dict.received_timestamp = rece_timestamp;
        data_dict.N = N;
        data_dict.D = D;
        data_dict.instances.reserve(N);
        data_dict.centroids.reserve(N);
        data_dict.features_vec.reserve(N);

        for(int i=0;i<N;i++){
            data_dict.nodes.emplace_back((uint32_t)i);
            data_dict.instances.emplace_back((uint32_t)msg->instances[i]);
            Eigen::Vector3d pt(msg->centroids[i].x,msg->centroids[i].y,msg->centroids[i].z);
            data_dict.centroids.emplace_back(pt);
            std::vector<float> feat_vec(msg->features.data.begin()+i*D,msg->features.data.begin()+(i+1)*D);
            data_dict.features_vec.emplace_back(feat_vec);
            // std::copy(feat_vec.begin(), feat_vec.end(), received_array[i]);
        }

        std::cout<<"receive feature array length "<<N<<" x "<<D <<" at timestamp "<<rece_timestamp<<"\n";

        // Log
        log.checksum = true;
        sub_logs.push_back(log);
    }
    */
    
    void Communication::dense_graph_callback(const sgloop_ros::DenseGraph::ConstPtr &msg, std::string agent_name)
    {
        int N = msg->nodes_number;
        int X = msg->points_number;
        int D = msg->features.layout.dim[1].size;        
        ROS_WARN("Received dense graph msg from agent %s. %d nodes and %d points \n",
                agent_name.c_str(), N, X);

        int recv_frame_id = msg->header.seq;
        float rece_timestamp = msg->header.stamp.toSec();
        Log log;
        log.direction="sub";
        if (X>0) log.msg_type="DenseGraph";
        else log.msg_type="CoarseGraph";
        log.frame_id = msg->header.seq;
        log.timestamp = rece_timestamp;

        // Check
        if(msg->instances.size() != msg->centroids.size() 
            || msg->instances.size()!=N 
            ||msg->features.data.size()!=N*D){
            ROS_WARN("Error: inconsistent instance and centroid size. Drop it.\n");
            log.checksum = false;
            sub_logs.push_back(log);
            return;
        }

        // Read
        assert(agents_data_dict.find(agent_name) != agents_data_dict.end());
        AgentDataDict &data_dict = agents_data_dict[agent_name];
        data_dict.clear();
        data_dict.frame_id = recv_frame_id;
        data_dict.received_timestamp = rece_timestamp;
        data_dict.N = N;
        data_dict.X = X;
        data_dict.D = D;
        data_dict.nodes.reserve(N);
        data_dict.instances.reserve(N);
        data_dict.centroids.reserve(N);
        data_dict.features_vec.reserve(N);
        data_dict.xyz.reserve(X);
        data_dict.labels.reserve(X);

        // Read coarse data
        for(int i=0;i<N;i++){ // nodes
            data_dict.nodes.emplace_back((uint32_t)msg->nodes[i]);
            data_dict.instances.emplace_back((uint32_t)msg->instances[i]);
            Eigen::Vector3d pt(msg->centroids[i].x,msg->centroids[i].y,msg->centroids[i].z);
            data_dict.centroids.emplace_back(pt);
            std::vector<float> feat_vec(msg->features.data.begin()+i*D,msg->features.data.begin()+(i+1)*D);
            data_dict.features_vec.emplace_back(feat_vec);
        }
        
        // std::cout<<"receive feature array length "<<N<<" x "<<D <<" at timestamp "<<rece_timestamp<<"\n";

        // Read dense data
        for(int k=0;k<X;k++){
            Eigen::Vector3d pt(msg->points[k].x,msg->points[k].y,msg->points[k].z);
            data_dict.xyz.emplace_back(pt);
            data_dict.labels.emplace_back((uint32_t)msg->labels[k]);
        }
        // std::cout<<"Receive "<< X<<" xyzi\n";

        // Log
        log.checksum = true;
        log.N = N;
        log.X = X;
        sub_logs.push_back(log);

    }

    void Communication::request_dense_callback(const std_msgs::Int32::ConstPtr &msg)
    {
        ROS_WARN("%s Received request dense msg\n", local_agent.c_str());
        if(msg->data>0) pub_dense_msg = true;
    }
    
    bool Communication::send_request_dense(const std::string &target_agent_name)
    {
        if(pub_agents_request_dense.find(target_agent_name) == pub_agents_request_dense.end()){
            ROS_WARN("Error: agent %s not found.\n",target_agent_name.c_str());
            return false;
        }
        else{
            std_msgs::Int32 req_msg;
            req_msg.data = 10;
            pub_agents_request_dense[target_agent_name].publish(req_msg);
            // ROS_WARN("Send request dense msg to %s\n",target_agent_name.c_str());
            return true;
        }
    }

    const AgentDataDict& Communication::get_remote_agent_data(const std::string agent_name)const
    {
        if(agents_data_dict.find(agent_name) == agents_data_dict.end()){
            ROS_WARN("Error: agent %s not found.\n",agent_name.c_str());
            return AgentDataDict{}; // return empty dict
        }
        else{
            return agents_data_dict.at(agent_name);
        } 
    }

    bool Communication::write_logs(const std::string &out_dir)
    {
        std::string pub_log_file = out_dir + "/pub_logs.txt";
        std::string sub_log_file = out_dir + "/sub_logs.txt";

        std::ofstream pub_log_stream;
        std::ofstream sub_log_stream;
        pub_log_stream.open(pub_log_file);
        sub_log_stream.open(sub_log_file);
        std::string header="# frame_id timestamp direction msg_type nodes_number points_number checksum\n";

        if(pub_log_stream.is_open() && sub_log_stream.is_open()){
            pub_log_stream<<header;
            for(const Log &log: pub_logs){
                pub_log_stream<<log.frame_id<<" "<<log.timestamp<<" "<<log.direction<<" "<<log.msg_type<<" "<<log.N<<" "<<log.X<<" ";
                if(log.checksum) pub_log_stream<<"true\n";
                else pub_log_stream<<"false\n";
            }

            sub_log_stream<<header;
            for(const Log &log: sub_logs){
                sub_log_stream<<log.frame_id<<" "<<log.timestamp<<" "<<log.direction<<" "<<log.msg_type<<" "<<log.N<<" "<<log.X<<" ";
                if(log.checksum) sub_log_stream<<"true\n";
                else sub_log_stream<<"false\n";
            }
            pub_log_stream.close();
            sub_log_stream.close();
            // std::cout<<"Write logs successfully.\n";
            return true;
        }
        else{
            ROS_WARN("Error: cannot open log files.\n");
            return false;
        }
    }
    
}

