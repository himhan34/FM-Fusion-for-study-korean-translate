#ifndef FMFUSION_DETECTION_H
#define FMFUSION_DETECTION_H

#include "opencv2/opencv.hpp"
#include "open3d/utility/IJsonConvertible.h"


namespace fmfusion
{
    
typedef std::pair<std::string,float> LabelScore;

struct BoundingBox{
    double u0,v0,u1,v1;
};

class Detection
{
public:
    Detection(const int id);
    Detection(const std::vector<LabelScore> &labels, const BoundingBox &bbox, const cv::Mat &instances_idxs): 
        labels_(labels), bbox_(bbox), instances_idxs_(instances_idxs) {};
    std::string extract_label_string() const;
    const cv::Point get_box_center(){return cv::Point((bbox_.u0+bbox_.u1)/2,(bbox_.v0+bbox_.v1)/2);};
    const int get_box_area(){return (bbox_.u1-bbox_.u0)*(bbox_.v1-bbox_.v0);}

public:
    unsigned int id_;
    std::vector<LabelScore> labels_;
    BoundingBox bbox_;
    cv::Mat instances_idxs_; // [H,W], CV_8UC1
};

typedef std::shared_ptr<Detection> DetectionPtr;

class DetectionFile: public open3d::utility::IJsonConvertible
{
public:
    DetectionFile(int min_mask, int max_box_area): min_mask_(min_mask), max_box_area_(max_box_area) {};

    bool updateInstanceMap(const std::string &instance_file);    

public:
    bool ConvertToJsonValue(Json::Value &value) const override {return true;};
    bool ConvertFromJsonValue(const Json::Value &value) override;

protected:
    std::vector<std::string> split_str(const std::string s, const std::string delim);
    int min_mask_;
    int max_box_area_;

public:
    std::vector<DetectionPtr> detections;
    std::vector<std::string> raw_tags;
    std::vector<std::string> filter_tags;
};

}

#endif //FMFUSION_DETECTION_H