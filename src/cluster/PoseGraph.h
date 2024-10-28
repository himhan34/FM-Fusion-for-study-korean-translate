#ifndef FMFUSION_POSEGRAPH_H
#define FMFUSION_POSEGRAPH_H

#include <unordered_map>
#include "open3d/Open3D.h"
#include "open3d/utility/IJsonConvertible.h"

namespace fmfusion
{

class PoseGraphFile: public open3d::utility::IJsonConvertible
{
public:
    PoseGraphFile() {};

public:
    bool ConvertToJsonValue(Json::Value &value) const override {return true;};
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    std::unordered_map<std::string, Eigen::Matrix4d_u> poses_;

};

}
#endif