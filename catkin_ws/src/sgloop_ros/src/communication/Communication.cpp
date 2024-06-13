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
                                std::string master_agent, std::vector<std::string> other_agents)
    {
        std::string pub_topic = master_agent + "/coarse_graph";
        pub_coarse_sg = nh.advertise<sgloop_ros::CoarseGraph>(pub_topic, 1000);
        std::cout<<"Init com for master agent: "<<master_agent<<"\n";

        for(const std::string &agent: other_agents){
            std::string sub_topic = agent + "/coarse_graph";
            callback bounded_callback = boost::bind(&Communication::coarse_graph_callback, this, _1, agent);
            ros::Subscriber listener = nh.subscribe(sub_topic, 1000, bounded_callback);
                // sub_topic, 1000, &Communication::coarse_graph_callback, this);
            agents_subscriber.push_back(listener);
            agents_data_dict[agent] = AgentDataDict{};
            std::cout<<"Init com with "<<agent<<"\n";
        }

        last_exchange_frame_id = -1;
        frame_duration = 0.1;
        time_offset = 12000;
    }

    
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
        graph_msg.header.frame_id = "none";

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

    const AgentDataDict& Communication::get_remote_agent_data(const std::string agent_name)const
    {
        if(agents_data_dict.find(agent_name) == agents_data_dict.end()){
            ROS_WARN("Error: agent %s not found.\n",agent_name.c_str());
            return AgentDataDict{}; // return empty dict
        }
        else{
            return agents_data_dict.at(agent_name);
        } 
        

        // queried_data_dict = agents_data_dict.at(agent_name);
        // return queried_data_dict.frame_id;
    }

    bool Communication::write_logs(const std::string &out_dir)
    {
        std::string pub_log_file = out_dir + "/pub_logs.txt";
        std::string sub_log_file = out_dir + "/sub_logs.txt";

        std::ofstream pub_log_stream;
        std::ofstream sub_log_stream;
        pub_log_stream.open(pub_log_file);
        sub_log_stream.open(sub_log_file);
        std::string header="# frame_id timestamp direction msg_type checksum\n";

        if(pub_log_stream.is_open() && sub_log_stream.is_open()){
            pub_log_stream<<header;
            for(const Log &log: pub_logs){
                pub_log_stream<<log.frame_id<<" "<<log.timestamp<<" "<<log.direction<<" "<<log.msg_type<<" ";//<<log.checksum<<"\n";
                if(log.checksum) pub_log_stream<<"true\n";
                else pub_log_stream<<"false\n";
            }

            sub_log_stream<<header;
            for(const Log &log: sub_logs){
                sub_log_stream<<log.frame_id<<" "<<log.timestamp<<" "<<log.direction<<" "<<log.msg_type<<" "; //<<log.checksum<<"\n";
                if(log.checksum) sub_log_stream<<"true\n";
                else sub_log_stream<<"false\n";
            }
            pub_log_stream.close();
            sub_log_stream.close();
            std::cout<<"Write logs successfully.\n";
            return true;
        }
        else{
            ROS_WARN("Error: cannot open log files.\n");
            return false;
        }
    }
    
}

