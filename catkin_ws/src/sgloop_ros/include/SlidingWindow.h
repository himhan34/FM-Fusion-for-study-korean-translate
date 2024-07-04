#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <eigen3/Eigen/Dense>

namespace sgloop_ros
{
    class SlidingWindow
    {
    private:
        float translation_threshold;
        std::vector<int> frame_ids;
        std::vector<float> translations;
        float total_translation;

        Eigen::Matrix4d last_pose;

    public:
        SlidingWindow(float translation_threshold_ = 100.0);
        ~SlidingWindow(){};

        void update_translation(const int &frame_id, const Eigen::Matrix4d &cur_pose); 
                            // float translation);

        int get_window_start_frame();

    };
    
    SlidingWindow::SlidingWindow(float translation_threshold_): 
        translation_threshold(translation_threshold_), total_translation(0.0)
    {

    }

    void SlidingWindow::update_translation(const int &frame_id, const Eigen::Matrix4d &cur_pose)
    {
        float translation = (cur_pose.block<3,1>(0,3) - last_pose.block<3,1>(0,3)).norm();

        translations.push_back(translation);
        frame_ids.push_back(frame_id);
        total_translation += translation;

        if(total_translation > translation_threshold)
        { // remove frames from the start
            while(total_translation > translation_threshold)
            {
                total_translation -= translations[0];
                frame_ids.erase(frame_ids.begin());
                translations.erase(translations.begin());
            }
        }

        last_pose = cur_pose;
    }

    int SlidingWindow::get_window_start_frame()
    {
        int start_frame;
        if(frame_ids.empty()) start_frame = 0;
        else start_frame = std::max(0,frame_ids[0]-1);

        // std::cout<<"start_frame: "<<start_frame<<", "
        //         <<"translation: " <<total_translation <<"\n";
        return start_frame;
    }
    
    
}

