#include "json/json.h"
#include "Detection.h"


namespace fmfusion
{

Detection::Detection(const int id): id_(id)
{

}

std::string Detection::extract_label_string() const
{
    std::stringstream ss;
    for(auto label_score:labels_){
        ss<<label_score.first<<"("<<std::fixed<<std::setprecision(2)<<label_score.second<<"), ";
        // <<label_score.second<<"), ";
    }
    return ss.str();
}

std::vector<std::string> DetectionFile::split_str(
    const std::string s, const std::string delim) 
{
    std::vector<std::string> list;
    auto start = 0U;
    auto end = s.find(delim);
    while (true) {
        list.push_back(s.substr(start, end - start));
        if (end == std::string::npos)
            break;
        start = end + delim.length();
        end = s.find(delim, start);
    }
    return list;
}

bool DetectionFile::ConvertFromJsonValue(const Json::Value &value)
{
    using namespace open3d;
    
    if (!value.isObject()) {
        std::cerr<<"DetectionField read JSON failed: unsupported json format."<<std::endl;
        // open3d::utility::LogWarning(
        //         "DetectionField read JSON failed: unsupported json "
        //         "format.");
        return false;
    }
    const Json::Value object_array = value["mask"];

    if (!object_array.isArray() || object_array.size() <2) {
        std::cerr<<"DetectionField cannot find valid objects."<<std::endl;
        return false;
    }
    
    // open3d::utility::LogInfo("DetectionField read {:d} objects.", object_array.size()-1);
    std::stringstream msg;

    for(int i=0;i<object_array.size();++i){ // update detected objects
        const Json::Value object = object_array[i];
        int detection_id = object["value"].asInt();
        if(detection_id==0) continue;
        auto detection = std::make_shared<Detection>(detection_id);

        auto label_score_map = object["labels"]; // {label:score}
        msg<<detection_id<<": ";
        for(auto it=label_score_map.begin();it!=label_score_map.end();++it){
            std::string label = it.key().asString();
            float score = (*it).asFloat();
            detection->labels_.push_back(std::make_pair(label,score));
            msg<<"("<<label<<","<<score<<") ";
        }

        auto box_array = object["box"];
        detection->bbox_.u0 = box_array[0].asDouble();
        detection->bbox_.v0 = box_array[1].asDouble();
        detection->bbox_.u1 = box_array[2].asDouble();
        detection->bbox_.v1 = box_array[3].asDouble();

        // update
        // int box_area = (detection->bbox_.u1-detection->bbox_.u0)*(detection->bbox_.v1-detection->bbox_.v0);
        // if(box_area<max_box_area_){
        detections.push_back(detection);
        msg<<"bbox("<<detection->bbox_.u0<<","<<detection->bbox_.v0<<","<<detection->bbox_.u1<<","<<detection->bbox_.v1<<") ";
        msg<<"\n";
        // }
    }

    // update raw tags
    auto raw_tag_array = split_str(value["raw_tags"].asString(), ".");
    msg<<"raw tags (";
    for(auto tag:raw_tag_array){
        raw_tags.push_back(tag.substr(tag.find_first_not_of(" ")));
        msg<<"$"<<tag.substr(tag.find_first_not_of(" "))<<"$";
    }
    msg<<")\n";

    // update filter tags
    auto filter_tag_array = split_str(value["tags"].asString(), ".");
    msg<<"filter tags (";
    for(auto tag:filter_tag_array){
        filter_tags.push_back(tag.substr(tag.find_first_not_of(" ")));
        msg<<"$"<<tag.substr(tag.find_first_not_of(" "))<<"$";
    }
    msg<<")\n";

    // for debug
    // std::cout<<msg.str()<<endl;

    return true;
}

bool DetectionFile::updateInstanceMap(const std::string &instance_file)
{
    const int K = detections.size();
    cv::Mat detection_map = cv::imread(instance_file, -1);
    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(detection_map, &min, &max, &min_loc, &max_loc);

    // todo: run sam with larger number of boxes
    if(max!=K){
        std::cerr<<"instance map has different number of instances"<<std::endl;
        return false;
    }
    assert (max == K), "instance map has different number of instances";
    std::vector<int> to_remove_detections;

    for(int idx=1;idx<=K;++idx){
        cv::Mat mask = (detection_map == idx);
        assert (detections[idx-1]->id_ == idx), "detection id is not consistent";
        detections[idx-1]->instances_idxs_ = mask; // [H,W], CV_8UC1
        if(cv::countNonZero(mask)<min_mask_ || detections[idx-1]->get_box_area()>max_box_area_){
            to_remove_detections.push_back(idx-1);
        }
    }

    // remove invalid detections 
    if(to_remove_detections.size()>0){
        std::cerr<<"Remove "<<to_remove_detections.size()<<" invalid detections"<<std::endl;
        for(int i=to_remove_detections.size()-1;i>=0;--i){
            detections.erase(detections.begin()+to_remove_detections[i]);
        }
    }

    return true;
}

}
