#include "Edge.h"

namespace fmfusion
{
    bool construct_edges(const std::vector<InstancePtr> &instances,
                         std::vector<EdgePtr> &edges, float radius_ratio, bool involve_floor)
    {
        std::string floor_names = "floor. carpet.";
        std::set<int> floors, ceilings;
        const int N = instances.size();
        std::stringstream msg;
        std::stringstream edge_msgs;

        // Object to Object
        for (int i=0; i<N; i++){
            InstancePtr src = instances[i];
            std::string src_label = src->get_predicted_class().first;
            float radius_src;
            if (floor_names.find(src_label) != std::string::npos){
                floors.emplace(i);
                continue;}
            if (src_label.find("wall")!=std::string::npos)
                radius_src = 0.5;
            else
                radius_src = src->min_box->extent_.norm()/2.0;
            msg<<src_label <<"("<<radius_src<<"): ";
            for (int j=i+1; j<N; j++){
                InstancePtr ref = instances[j];
                std::string ref_label = ref->get_predicted_class().first;
                if(src->id_ == ref->id_ || floor_names.find(ref_label) != std::string::npos)
                    continue;
                msg<<ref_label<<"";
                float radius_ref;
                if (ref_label.find("wall")!=std::string::npos)
                    radius_ref = 0.5;
                else
                    radius_ref = ref->min_box->extent_.norm()/2.0;
                float search_radius = radius_ratio * std::max(radius_src, radius_ref);
                float dist = (src->centroid - ref->centroid).norm();
                msg<<"("<<radius_ref<<")("<<search_radius<<"),";
                if(dist<search_radius){
                    EdgePtr edge = std::make_shared<Edge>(src->id_, ref->id_);
                    edges.push_back(edge);
                    edge_msgs<<"("<<src->id_<<","<<ref->id_<<"),";
                }
            }
            msg<<"\n";
        }
        edge_msgs<<"\n";
        std::cout<<edge_msgs.str();
        // std::cout<<msg.str();

        std::cout<<"Constructed "<<edges.size()<<" edges.\n";
        if (floors.empty() || !involve_floor){
            return true;
        }

        // Object to one floor
        std::cout<<"Constructing edges for "<< floors.size()<<" floors.\n";
        for (int i=0; i<N; i++){
            if (floors.find(i)!=floors.end()) continue;
            std::pair<InstanceId,float> closet_floor = std::make_pair(-1,1000000.0);
            InstancePtr src = instances[i];
            for (int floor_index:floors){
                float dist = (src->centroid - instances[floor_index]->centroid).norm();
                if (dist<closet_floor.second)
                    closet_floor = std::make_pair(instances[floor_index]->id_,dist);
            }

            //
            if(closet_floor.first>=0){ // valid edge
                EdgePtr edge = std::make_shared<Edge>(src->id_, closet_floor.first);
                edges.push_back(edge);
            }
        }


        //
        std::cout<<"Constructed "<<edges.size()<<" edges.\n";
        return true;
    }

}
