#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <map>
#include <fstream>
#include <torch/torch.h>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sgloop_ros/CoarseGraph.h>
#include <eigen3/Eigen/Dense>

namespace SgCom
{
    struct Log
    {
        int frame_id;
        float timestamp;
        std::string direction; // "pub" or "sub"
        std::string msg_type; // "CoarseGraph", "DenseGraph"
        bool checksum;
    };
    
    struct AgentDataDict{
        int frame_id=-1;
        float received_timestamp;
        int N=0;
        int D=0; // coarse node feature dimension
        std::vector<uint32_t> instances;
        std::vector<Eigen::Vector3d> centroids;
        std::vector<std::vector<float>> features_vec;

        void clear(){
            instances.clear();
            centroids.clear();
            features_vec.clear();
            N=0;
            D=0;
        };

        std::string print_msg()const{
            std::stringstream ss;
            for(int i=0;i<N;i++){
                ss<<"instance: "<<instances[i]<<" centroid: "<<centroids[i].transpose();
                ss<<"\n";
            }
            return ss.str();
        }

    };

    typedef const boost::function<void(const sgloop_ros::CoarseGraphConstPtr &)>  callback;

    void convert_tensor(const std::vector<std::vector<float>> &features_vector, 
                    const int &N, const int &D, torch::Tensor &features_tensor);
                    
    class Communication
    {
        public:
            /// @brief Each MASTER_AGENT publish their features to "MASTER_AGENT/graph_topics"
            ///        And it subscribes to "OTHER_AGENT/graph_topics" for other agents' features.
            /// @param nh 
            /// @param nh_private 
            /// @param master_agent 
            /// @param other_agents 
            Communication(ros::NodeHandle &nh, ros::NodeHandle &nh_private,
                        std::string master_agent="agent_a", std::vector<std::string> other_agents={});
            ~Communication(){};

            bool broadcast_coarse_graph(int frame_id,
                                const std::vector<uint32_t> &instances,
                                const std::vector<Eigen::Vector3d> &centroids,
                                const int& N, const int &D,
                                const std::vector<std::vector<float>> &features);

            void coarse_graph_callback(const sgloop_ros::CoarseGraph::ConstPtr &msg,std::string agent_name);

            const AgentDataDict& get_remote_agent_data(const std::string agent_name)const;
            // AgentDataDict &queried_data_dict)const;

            bool write_logs(const std::string &out_dir);

        private:

            float frame_duration; // in seconds
            float time_offset; // in seconds

            ros::Publisher pub_coarse_sg;
            std::vector<ros::Subscriber >agents_subscriber;
            std::map<std::string,AgentDataDict> agents_data_dict;

            std::vector<Log> pub_logs;
            std::vector<Log> sub_logs;

            int last_exchange_frame_id; // At an exchange frame, the latest agent data dict is exported.

    };
    
    

}


#endif // COMMUNICATION_H

