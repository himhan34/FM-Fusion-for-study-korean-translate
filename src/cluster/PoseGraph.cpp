#include "json/json.h"
#include "PoseGraph.h"

namespace fmfusion
{

bool PoseGraphFile::ConvertFromJsonValue(const Json::Value &value)
{
    using namespace open3d;
    
    if (!value.isObject()) {
        open3d::utility::LogWarning(
                "PoseGraph read JSON failed: unsupported json "
                "format.");
        return false;
    }
    const Json::Value scans_pose = value;
    // std::cout << scans_pose << std::endl;

    if (scans_pose.size() <1) {
        open3d::utility::LogWarning(
                "PoseGraph cannot find valid objects.");
        return false;
    }
    

    for (auto it=scans_pose.begin();it!=scans_pose.end();it++){
        std::string scene_name = it.key().asString();
        Eigen::Matrix4d_u T;
        for(int i=0;i<4;++i){
            for(int j=0;j<4;++j){
                T(i,j) = (*it)[i*4+j].asDouble();
            }
        }
        poses_[scene_name] = T;
        // std::cout<<scene_name<<"\n";
        // std::cout<<T<<"\n";

    }

    std::cout<<"Read "<<poses_.size()<<" poses.\n";
    return true;
}



}

