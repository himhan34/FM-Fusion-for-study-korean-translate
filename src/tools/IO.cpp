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

    bool read_transformation(const std::string &transformation_file, 
                        Eigen::Matrix4d &transformation)
    {
        using namespace std;
        fstream file(transformation_file,fstream::in);
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
                    transformation(i,j) = stod(substr);
                    j++;
                }
                i++;
            }
            file.close();
            std::cout<<"Read gt transformation from "<<transformation_file<<":\n";
            return true;
        }
        else
            return false;
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

    /// Read rgb from color, depth from depth and pose from pose
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
        using namespace open3d::utility::filesystem;
        int max_frames = 6000;

        int index =0;
        char buffer[DEFAULT_IO_BUFFER_SIZE];
        std::string da_dir = JoinPath({root_dir,association_name});
        std::string trajectory_dir = JoinPath({root_dir,trajectory_name});

        auto camera_trajectory = open3d::io::CreatePinholeCameraTrajectoryFromFile(trajectory_dir);
        FILE *file = FOpen(da_dir, "r");
        if (file == NULL) {
            open3d::utility::LogWarning("Unable to open file {}", da_dir);
            fclose(file);
            return false;
        }
        while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file) && index<max_frames) {
            std::vector<std::string> st = open3d::utility::SplitString(buffer, "\t\r\n ");
            if (st.size() >= 2) {
                std::string depth_file = JoinPath({root_dir, st[0]});
                std::string color_file = JoinPath({root_dir, st[1]});
                if (FileExists(depth_file) && FileExists(color_file)) {
                    RGBDFrameDirs frame_dirs = std::make_pair(color_file,depth_file);
                    pose_table.emplace_back(camera_trajectory->parameters_[index].extrinsic_.inverse().cast<double>());
                    rgbd_table.emplace_back(frame_dirs);
                }
                index++;
            }
        }
        fclose(file);
        open3d::utility::LogWarning("Read {:d} RGB-D frames with poses",rgbd_table.size());
        return true;
    }

    void extract_match_instances(const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            const std::vector<fmfusion::NodePtr> &src_nodes,
                            const std::vector<fmfusion::NodePtr> &ref_nodes,
                            std::vector<std::pair<fmfusion::InstanceId,fmfusion::InstanceId>> &match_instances)
    {
        int Ns = src_nodes.size();
        int Nr = ref_nodes.size();
        for (auto pair: match_pairs){
            assert(pair.first<Ns && pair.second<Nr), "Matched node index out of range.";
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
            // msg<<"("<<src_node->instance_id<<","<<ref_node->instance_id<<") "
            // <<"("<<src_node->semantic<<","<<ref_node->semantic<<")\n";
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


    bool save_match_results(const float &timestamp,
                            const Eigen::Matrix4d &pose,
                            const std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            const std::vector<Eigen::Vector3d> &src_centroids,
                            const std::vector<Eigen::Vector3d> &ref_centroids,
                            const std::string &output_file_dir)
    {
        std::ofstream file(output_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_file_dir<<std::endl;
            return false;
        }

        file<<"# timetstamp: "<<timestamp<<"; pose\n";
        file<<std::fixed<<std::setprecision(6);
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                file<<pose(i,j)<<" ";
            }
            file<<std::endl;
        }

        file<<"# src, ref, src_centroid, ref_centroid\n";
        file<<std::fixed<<std::setprecision(3);
        for (size_t i=0; i<match_pairs.size(); i++){
            file<<"("<<match_pairs[i].first<<","<<match_pairs[i].second<<") "
                <<std::fixed<<std::setprecision(3)
                <<src_centroids[i][0]<<" "<<src_centroids[i][1]<<" "<<src_centroids[i][2]<<" "
                <<ref_centroids[i][0]<<" "<<ref_centroids[i][1]<<" "<<ref_centroids[i][2]<<std::endl;
            // <<src_centroids[i].transpose()<<" "<<ref_centroids[i].transpose()<<std::endl;
        }

        file.close();
        return true;
    };
    

    bool load_match_results(const std::string &match_file_dir,
                            float &timestamp,
                            Eigen::Matrix4d &pose,
                            std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                            std::vector<Eigen::Vector3d> &src_centroids,
                            std::vector<Eigen::Vector3d> &ref_centroids,
                            bool verbose)
    {
        std::ifstream file(match_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<match_file_dir<<std::endl;
            return false;
        }

        std::string line;
        std::getline(file,line);

        if(line.find("# timetstamp")!=std::string::npos){

            timestamp = std::stof(line.substr(line.find(":")+1, line.find(";")));

            for(int i=0; i<4; i++){
                std::getline(file,line);
                std::stringstream ss(line);
                for(int j=0; j<4; j++){
                    ss>>pose(i,j);
                }
            }
            if(verbose)
                std::cout<<"Load timestamp: "<<timestamp<<"\n"
                        <<"Load pose: \n"<<pose<<"\n";
        }

        while(std::getline(file,line)){
            if(line.find("# src, ref, src_centroid, ref_centroid")!=std::string::npos){
                while(std::getline(file,line)){
                    std::stringstream ss(line);
                    std::string substr;
                    ss>>substr;
                    if(substr.find("(")!=std::string::npos){
                        substr = substr.substr(1,substr.size()-2);
                        auto pair = fmfusion::utility::split_str(substr,",");
                        int src_id = stoi(pair[0]);
                        int ref_id = stoi(pair[1]);
                        Eigen::Vector3d src_centroid, ref_centroid;
                        ss>>src_centroid[0]>>src_centroid[1]>>src_centroid[2];
                        ss>>ref_centroid[0]>>ref_centroid[1]>>ref_centroid[2];

                        match_pairs.push_back(std::make_pair(src_id,ref_id));
                        src_centroids.push_back(src_centroid);
                        ref_centroids.push_back(ref_centroid);

                        if(verbose)
                            std::cout<<"("<<src_id<<","<<ref_id<<") "
                                <<src_centroid.transpose()<<" "<<ref_centroid.transpose()<<"\n";
                    }
                }
            }
        }

        if(verbose)
            std::cout<<"Load "<<src_centroids.size()<<" correspondences.\n";

        file.close();
        return true;
    };

    bool save_pose(const std::string &output_dir, const Eigen::Matrix4d &pose)
    {
        std::ofstream file(output_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_dir<<std::endl;
            return false;
        }

        file<<std::fixed<<std::setprecision(6);
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                file<<pose(i,j)<<" ";
            }
            file<<std::endl;
        }

        file.close();
        return true;
    };

    bool save_corrs_match_indices(const std::vector<int> &corrs_match_indices,
                                const std::vector<float> &corrs_match_scores,
                                const std::string &output_file_dir)
    {
        std::ofstream file(output_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_file_dir<<std::endl;
            return false;
        }

        file<<std::fixed<<std::setprecision(2);
        for(int i=0; i<corrs_match_indices.size(); i++){
            file<<corrs_match_indices[i]<<" "<<corrs_match_scores[i]<<std::endl;
        }

        file.close();
        return true;
    };

    bool load_corrs_match_indices(const std::string &corrs_match_indices_file,
                                std::vector<int> &corrs_match_indices,
                                std::vector<float> &corrs_match_scores)
    {
        std::ifstream file(corrs_match_indices_file);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<corrs_match_indices_file<<std::endl;
            return false;
        }

        std::string line;
        while(std::getline(file,line)){
            std::stringstream ss(line);
            int index;
            float score;
            ss>>index>>score;
            corrs_match_indices.push_back(index);
            corrs_match_scores.push_back(score);
            // corrs_match_indices.push_back(std::stoi(line));
        }

        std::cout<<"Load "<<corrs_match_indices.size()<<" correspodences match index.\n";

        file.close();
        return true;
    };

    bool load_node_matches(const std::string &match_file_dir,
                        std::vector<std::pair<uint32_t,uint32_t>> &match_pairs,
                        std::vector<bool> &match_tp_masks,
                        std::vector<Eigen::Vector3d> &src_centroids,
                        std::vector<Eigen::Vector3d> &ref_centroids,
                        bool verbose)
    {
        std::ifstream file(match_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<match_file_dir<<std::endl;
            return false;
        }

        std::string line;
        while(std::getline(file,line)){
            int src_id, ref_id;
            int tp_mask;
            Eigen::Vector3d src_centroid, ref_centroid;
            if(line.find("#")!=std::string::npos) continue;

            std::stringstream ss(line);
            ss>>src_id>>ref_id>>tp_mask;
            ss>>src_centroid[0]>>src_centroid[1]>>src_centroid[2];
            ss>>ref_centroid[0]>>ref_centroid[1]>>ref_centroid[2];

            match_pairs.push_back(std::make_pair(src_id,ref_id));
            if (tp_mask==1) match_tp_masks.push_back(true);
            else match_tp_masks.push_back(false);
            src_centroids.push_back(src_centroid);
            ref_centroids.push_back(ref_centroid);
        }

        std::cout<<"Load "<<src_centroids.size()<<" correspondences.\n";

        return true;
    }
    
    bool load_single_col_mask(const std::string &mask_file_dir, std::vector<bool> &mask)
    {
        std::ifstream file(mask_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<mask_file_dir<<std::endl;
            return false;
        }

        std::string line;
        while(std::getline(file,line)){
            int mask_val;
            if(line.find("#")!=std::string::npos) continue;
            std::stringstream ss(line);
            ss>>mask_val;
            if (mask_val==1) mask.push_back(true);
            else mask.push_back(false);
        }

        std::cout<<"Load "<<mask.size()<<" mask values.\n";

        return true;
    }

    bool load_pose_file(const std::string &pose_file_dir, Eigen::Matrix4d &pose, bool verbose)
    {
        std::ifstream file(pose_file_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<pose_file_dir<<std::endl;
            return false;
        }

        std::string line;
        for(int i=0; i<4; i++){
            std::getline(file,line);
            std::stringstream ss(line);
            for(int j=0; j<4; j++){
                ss>>pose(i,j);
            }
        }
        if(verbose)
            std::cout<<"Load pose: \n"<<pose<<"\n";

        return true;
    }

    bool read_loop_transformations(const std::string &loop_file_dir,
                        std::vector<LoopPair> &loop_pairs,
                        std::vector<Eigen::Matrix4d> &loop_transformations)
    {
        std::ifstream loop_file(loop_file_dir);
        if (!loop_file.is_open()){
            std::cerr<<"Cannot open loop file: "<<loop_file_dir<<std::endl;
            return false;
        }

        std::string line;
        while (std::getline(loop_file, line)){
            if(line.find("#")!=std::string::npos){
                continue;
            }
            std::istringstream iss(line);
            Eigen::Vector3d t_vec;
            Eigen::Quaterniond quat;
            std::string src_frame, ref_frame;

            iss>>src_frame>>ref_frame;
            iss>>t_vec[0]>>t_vec[1]>>t_vec[2]>>quat.x()>>quat.y()>>quat.z()>>quat.w();

            Eigen::Matrix4d transformation;
            transformation.setIdentity();

            transformation.block<3,1>(0,3) = t_vec;
            transformation.block<3,3>(0,0) = quat.toRotationMatrix();
            // std::cout<<src_frame<<"->"<<ref_frame<<std::endl;

            loop_pairs.push_back(std::make_pair(src_frame, ref_frame));
            loop_transformations.push_back(transformation);
        }

        std::cout<<"Load "<<loop_transformations.size()<<" loop transformations"<<std::endl;

        return true;
    }

    bool read_loop_pairs(const std::string &loop_file_dir,
                        std::vector<LoopPair> &loop_pairs,
                        std::vector<bool> &loop_tp_masks)
    {
        std::ifstream loop_file(loop_file_dir);
        if (!loop_file.is_open()){
            std::cerr<<"Cannot open loop file: "<<loop_file_dir<<std::endl;
            return false;
        }

        int count_true = 0;
        std::string line;
        while (std::getline(loop_file, line)){
            if(line.find("#")!=std::string::npos){
                continue;
            }
            std::istringstream iss(line);
            std::string src_frame, ref_frame;
            int tp_mask;

            iss>>src_frame>>ref_frame>>tp_mask;

            loop_pairs.push_back(std::make_pair(src_frame, ref_frame));
            if (tp_mask==1) {
                loop_tp_masks.push_back(true);
                count_true++;
            }
            else loop_tp_masks.push_back(false);
        }

        std::cout<<"Load "<< count_true<<"/"
                    <<loop_pairs.size()<<" true loop pairs"<<std::endl;
        return true;

    }

    bool read_frames_poses(const std::string &frame_pose_file,
                    std::unordered_map<std::string, Eigen::Matrix4d> &frame_poses)
    {
        std::ifstream pose_file(frame_pose_file);
        if (!pose_file.is_open()){
            std::cerr<<"Cannot open pose file: "<<frame_pose_file<<std::endl;
            return false;
        }

        std::string line;
        while (std::getline(pose_file, line)){
            if(line.find("#")!=std::string::npos){
                continue;
            }
            std::istringstream iss(line);
            std::string frame_name;
            Eigen::Vector3d p;
            Eigen::Quaterniond q;
            Eigen::Matrix4d pose;
            iss>>frame_name;
            iss>>p[0]>>p[1]>>p[2]>>q.x()>>q.y()>>q.z()>>q.w();
            pose.setIdentity();
            pose.block<3,1>(0,3) = p;
            pose.block<3,3>(0,0) = q.toRotationMatrix();

            frame_poses[frame_name] = pose;
        }

        return true;
    }

    bool read_entire_camera_poses(const std::string &scene_folder,
                        std::unordered_map<std::string, Eigen::Matrix4d> &src_poses_map)
    {
        std::vector<RGBDFrameDirs> src_rgbds;
        std::vector<Eigen::Matrix4d> src_poses;

        bool read_ret = construct_preset_frame_table(
            scene_folder,"data_association.txt","trajectory.log",src_rgbds,src_poses);
        if(!read_ret) {
            return false;
        }

        for (int i=0;i<src_rgbds.size();i++){
            std::string frame_file_name = open3d::utility::filesystem::GetFileNameWithoutDirectory(src_rgbds[i].first);
            std::string frame_name = frame_file_name.substr(0, frame_file_name.size()-4);
            // std::cout<<frame_name<<std::endl;
            // src_poses_map[src_rgbds[i].first] = src_poses[i];
            src_poses_map[frame_name] = src_poses[i];
        }

        return true;
    }


    bool save_graph_centroids(const std::vector<Eigen::Vector3d> &src_centroids, 
                            const std::vector<Eigen::Vector3d> &ref_centroids,
                            const std::string &output_dir)
    {
        std::ofstream file(output_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_dir<<std::endl;
            return false;
        }

        file<<"# src graph: "<<src_centroids.size()<< " nodes\n";

        file<<std::fixed<<std::setprecision(3);
        for(int i=0; i<src_centroids.size(); i++){
            file<<src_centroids[i][0]<<" "<<src_centroids[i][1]<<" "<<src_centroids[i][2]<<std::endl;
        }

        file << "# ref graph: " << ref_centroids.size() << " nodes\n";
        for (int i = 0; i < ref_centroids.size(); i++) {
            file << ref_centroids[i][0] << " " << ref_centroids[i][1] << " " << ref_centroids[i][2] << std::endl;
        }

        file.close();
        return true;
    }

    bool save_instance_info(const std::vector<Eigen::Vector3d> &centroids,
                        const std::vector<std::string> &labels,
                        const std::string &output_dir)
    {
        std::ofstream file(output_dir);
        if (!file.is_open()){
            std::cerr<<"Failed to open file: "<<output_dir<<std::endl;
            return false;
        }

        file<<"# x, y, z, semantic\n";
        file<<std::fixed<<std::setprecision(3);
        for(int i=0; i<centroids.size(); i++){
            file<<centroids[i][0]<<", "<<centroids[i][1]<<", "<<centroids[i][2]<<", "
                <<labels[i]<<std::endl;
        }

        file.close();
        return true;
    }

    bool load_instance_info(const std::string &instance_info_file,
                        std::vector<Eigen::Vector3d> &centroids,
                        std::vector<std::string> &labels)
    {
        std::ifstream file(instance_info_file);
        if (!file.is_open()){
            return false;
        }

        std::string line;
        while(std::getline(file,line)){
            if(line.find("#")!=std::string::npos) continue;
            Eigen::Vector3d centroid;
            std::string label;
            auto parts = utility::split_str(line,",");
            centroid[0] = std::stod(parts[0]);
            centroid[1] = std::stod(parts[1]);
            centroid[2] = std::stod(parts[2]);
            label = parts[3];
            centroids.push_back(centroid);
            labels.push_back(label.substr(1));
            // std::cout<<centroid.transpose()<<" "<<label<<std::endl;
        }

        // std::cout<<"Load "<<centroids.size()<<" instance info.\n";

        return true;
    }

    bool write_time(const std::vector<std::string> & header,
                const std::vector<double> & time,
                const std::string & file_name)
    {
        std::ofstream file(file_name);
        if (!file.is_open()) {
            return false;
        }
        // header
        for (int i = 0; i < header.size(); i++) 
            file << "# " << header[i] << " ";
        file << std::endl;

        // data
        for (int i = 0; i < time.size(); i++) 
            file << time[i] << " ";
        file << std::endl;

        file.close();
        return true;
    }

} // namespace name



}
