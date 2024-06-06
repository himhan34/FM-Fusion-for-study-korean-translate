// #include "Tools.h"
#include "IO.h"

namespace fmfusion
{

namespace IO
{
    bool read_rs_intrinsic(const std::string intrinsic_dir, 
                        open3d::camera::PinholeCameraIntrinsic &intrinsic_)
    {
        using namespace std;
        open3d::utility::LogInfo("Read intrinsic {:s}",intrinsic_dir);
        fstream file(intrinsic_dir,fstream::in);
        double fx,fy,cx,cy;
        int width, height;
        if(file.is_open()){
            string line{""};
            while(getline(file,line)){
                open3d::utility::LogInfo("{:s}",line);
                auto parts = fmfusion::utility::split_str(line,"=");
                if(parts[0].find("color_width")!=string::npos)
                    width = stoi(parts[1].substr(1));
                else if (parts[0].find("color_height")!=string::npos)
                    height = stoi(parts[1].substr(1));
                else if (parts[0].find("color_fx")!=string::npos)
                    fx = stod(parts[1].substr(1));
                else if ((parts[0].find("color_fy")!=string::npos))
                    fy = stod(parts[1].substr(1));
                else if ((parts[0].find("color_cx")!=string::npos))
                    cx = stod(parts[1].substr(1));
                else if ((parts[0].find("color_cy")!=string::npos))
                    cy = stod(parts[1].substr(1));
                    // utility::LogInfo("Intrinsic: {:s}",parts[1].substr(1));
            }
            file.close();
            std::cout<<fx<<","<<fy<<"\n";
            // open3d::utility::LogInfo("{:d},{:d},{:d},{:d}",fx,fy,cx,cy);
            intrinsic_.SetIntrinsics(width,height,fx,fy,cx,cy);
            return true;
        }
        return false;
    }

    bool read_scannet_intrinsic(const std::string intrinsic_folder,
        open3d::camera::PinholeCameraIntrinsic &intrinsic_)
    {
        using namespace std;
        open3d::utility::LogInfo("Read intrinsic {:s}",intrinsic_folder);
        fstream f_mat(intrinsic_folder+"/intrinsic_depth.txt",fstream::in);
        fstream f_shape(intrinsic_folder+"/sensor_shapes.txt",fstream::in);
        double fx,fy,cx,cy;
        int width, height;
        if(f_mat.is_open() && f_shape.is_open()){

            string line{""};
            string line0, line1; 
            string sline0,sline1,sline2, sline3;
            getline(f_mat,line0);
            getline(f_mat,line1);
            auto row0 = fmfusion::utility::split_str(line0," ");
            auto row1 = fmfusion::utility::split_str(line1," ");
            fx = stod(row0[0]); cx = stod(row0[2]);
            fy = stod(row1[1]); cy = stod(row1[2]);

            getline(f_shape,sline0);
            getline(f_shape,sline1);
            getline(f_shape,sline2);
            getline(f_shape,sline3);

            width = stoi(fmfusion::utility::split_str(sline2,":")[1]);
            height = stoi(fmfusion::utility::split_str(sline3,":")[1]);

            f_mat.close();
            f_shape.close();

            // std::cout<<width<<","<<height<<"\n";
            // std::cout<<fx<<","<<fy<<","<<cx<<","<<cy<<","<<"\n";
            intrinsic_.SetIntrinsics(width,height,fx,fy,cx,cy);
            return true;
        }
        else return false;

    }

    bool frames_srt_func(const std::string &a, const std::string &b)
    {
        std::string name_a = a.substr(a.find_last_of("/")+1); 
        std::string name_b = b.substr(b.find_last_of("/")+1); // frame-000000.png
        if(name_a.find("-")!=std::string::npos) 
            return stoi(name_a.substr(name_a.find_last_of("-")+1,name_a.find_last_of("."))) < stoi(name_b.substr(name_b.find_last_of("-")+1,name_b.find_last_of(".")));
        else
            return stoi(name_a.substr(0,name_a.find_last_of("."))) < stoi(name_b.substr(0,name_b.find_last_of(".")));
    }

    void construct_sorted_frame_table(const std::string &scene_dir,
        std::vector<RGBDFrameDirs> &frame_table, std::vector<Eigen::Matrix4d> &pose_table)
    {
        using namespace std;
        using namespace open3d::utility::filesystem;

        std::vector<std::string> rgb_frames, depth_frames;
        open3d::utility::filesystem::ListFilesInDirectory(scene_dir+"/color", rgb_frames);
        open3d::utility::filesystem::ListFilesInDirectory(scene_dir+"/depth", depth_frames);

        assert(rgb_frames.size()==depth_frames.size()),"rgb and depth frames should have the same size.";
        int K = rgb_frames.size();

        std::sort(rgb_frames.begin(),rgb_frames.end(),frames_srt_func);
        std::sort(depth_frames.begin(),depth_frames.end(),frames_srt_func);

        for(int k=0;k<K;k++){
            auto frame_name = rgb_frames[k].substr(rgb_frames[k].find_last_of("/")+1); // eg. frame-000000.png
            string pose_dir = JoinPath({scene_dir,"pose",frame_name.substr(0,frame_name.find_last_of("."))+".txt"});
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            fstream file(pose_dir,fstream::in);
            if(file.is_open()){
                string line{""};
                int i=0;
                while(getline(file,line)){
                    // std::cout<<line<<"\n";
                    std::stringstream ss(line);
                    int j=0;
                    while(ss.good() && j<4){
                        std::string substr;
                        getline(ss, substr, ' ');
                        if(substr.empty()) continue;
                        // cout<<substr<<",";
                        pose(i,j) = stod(substr);
                        j++;
                    }
                    // std::cout<<"\n";
                    i++;
                }
                // std::cout<<pose<<"\n";
                file.close();
            }
            else{
                open3d::utility::LogWarning("Failed to read pose file {:s}",pose_dir);
                continue;
            }

            // cout<<rgb_frames[k]<<endl;
            // cout<<depth_frames[k]<<endl;
            // cout<<pose<<endl;

            frame_table.emplace_back(rgb_frames[k],depth_frames[k]);
            pose_table.emplace_back(pose);
        }

        open3d::utility::LogInfo("Read {:d} frames",frame_table.size());

    }

    bool construct_preset_frame_table(const std::string &root_dir,
                                    const std::string &association_name,
                                    const std::string &trajectory_name,
                                    std::vector<RGBDFrameDirs> &rgbd_table,
                                    std::vector<Eigen::Matrix4d> &pose_table)
    {
        using namespace std;
        using namespace open3d::utility;
        int max_frames = 6000;

        int index =0;
        char buffer[DEFAULT_IO_BUFFER_SIZE];
        std::string da_dir = filesystem::JoinPath({root_dir,association_name});
        std::string trajectory_dir = filesystem::JoinPath({root_dir,trajectory_name});

        auto camera_trajectory = open3d::io::CreatePinholeCameraTrajectoryFromFile(trajectory_dir);
        FILE *file = filesystem::FOpen(da_dir, "r");
        if (file == NULL) {
            LogWarning("Unable to open file {}", da_dir);
            fclose(file);
            return false;
        }
        while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file) && index<max_frames) {
            std::vector<std::string> st = SplitString(buffer, "\t\r\n ");
            if (st.size() >= 2) {
                std::string depth_file = filesystem::JoinPath({root_dir, st[0]});
                std::string color_file = filesystem::JoinPath({root_dir, st[1]});
                if (filesystem::FileExists(depth_file) &&
                    filesystem::FileExists(color_file)) {
                    RGBDFrameDirs frame_dirs = std::make_pair(color_file,depth_file);
                    pose_table.emplace_back(camera_trajectory->parameters_[index].extrinsic_.inverse().cast<double>());
                    rgbd_table.emplace_back(frame_dirs);
                }
                index++;
            }
        }
        fclose(file);
        LogWarning("Read {:d} RGB-D frames with poses",rgbd_table.size());
        return true;
    }

    void extract_match_instances(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            const std::vector<fmfusion::NodePtr> &src_nodes,
                            const std::vector<fmfusion::NodePtr> &ref_nodes,
                            std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> &match_instances)
    {

        for (auto pair: match_pairs){
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            match_instances.push_back(std::make_pair(src_node->instance_id, ref_node->instance_id));
        }

    }

    void extract_instance_correspondences(const std::vector<fmfusion::NodePtr> &src_nodes, 
                                            const std::vector<fmfusion::NodePtr> &ref_nodes, 
                                            const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                                            const std::vector<float> &match_scores,
                                            std::vector<Eigen::Vector3d> &src_centroids, std::vector<Eigen::Vector3d> &ref_centroids)
    {
        std::stringstream msg;
        msg<<match_pairs.size()<<" Matched pairs: \n";

        for (auto pair: match_pairs){
            auto src_node = src_nodes[pair.first];
            auto ref_node = ref_nodes[pair.second];
            src_centroids.push_back(src_node->centroid);
            ref_centroids.push_back(ref_node->centroid);
            msg<<"("<<src_node->instance_id<<","<<ref_node->instance_id<<") "
            <<"("<<src_node->semantic<<","<<ref_node->semantic<<")\n";
        }

        // std::cout<<msg.str()<<std::endl;
    };

    bool save_match_results(const Eigen::Matrix4d &pose,
                            const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs, 
                            const std::vector<float> &match_scores,
                            const std::string &output_file_dir)
    {
        std::ofstream file(output_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_file_dir<<std::endl;
            return false;
        }

        file<<"# pose\n";
        file<<std::fixed<<std::setprecision(6);
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                file<<pose(i,j)<<" ";
            }
            file<<std::endl;
        }

        file<<"# src, ref, score\n";
        file<<std::fixed<<std::setprecision(3);
        for (size_t i=0; i<match_pairs.size(); i++){
            file<<"("<<match_pairs[i].first<<","<<match_pairs[i].second<<") "
            <<match_scores[i]<<std::endl;
        }

        file.close();
        return true;
    };

    
} // namespace name



}
