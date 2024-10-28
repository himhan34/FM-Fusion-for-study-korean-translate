#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>


namespace fmfusion
{
    class TicToc {
        public:
            TicToc() {
                tic();
            }

            void tic() {
                start = std::chrono::system_clock::now();
            }

            double toc(bool restart = false) {
                end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                if(restart) tic();
                return elapsed_seconds.count() * 1000;
            }

        private:
            std::chrono::time_point<std::chrono::system_clock> start, end;
    };

    class TicTocArray
    {
        TicTocArray() {
            tic_toc.tic();
        }

        double record(const std::string &name){
            double record = tic_toc.toc();
            tic_toc.tic();
            time_records.push_back(std::make_pair(name, record));
        }

        std::string print(){
            std::stringstream msg;
            for (auto record:time_records){
                msg<<record.first<<": "<<record.second<<"ms, ";
            }
            msg<<"\n";
            return msg.str();
        }

        public:
            TicToc tic_toc;
            std::vector<std::pair<std::string,double>> time_records;
    };

    class TicTocSequence
    {
        public:
            TicTocSequence(string header_,int K_=10):header(header_),K(K_){

            }

            void tic(){
                tic_toc.tic();
                frame_records.push_back(cur_frame);
                cur_frame.clear();
            };

            void toc(){
                float duration = tic_toc.toc();
                if(!cur_frame.empty()) duration -= cur_frame.back();
                cur_frame.push_back(duration);
            }

            void fill_zeros(int fill_segments=1){
                std::vector<float> zeros(fill_segments,0);
                cur_frame.insert(cur_frame.end(),zeros.begin(),zeros.end());
            }

            bool export_data(std::string output_dir){
                float sum_val[10]= {0,0,0,0,0,0,0,0,0,0};
                int count[10]={0,0,0,0,0,0,0,0,0,0};
                int num_segments=0;

                std::ofstream file;
                file.open(output_dir);
                if (!file.is_open()){
                    std::cout<<"Failed to open file: "<<output_dir<<std::endl;
                    return false;
                }
                file<<header<<std::endl;
                file<<std::fixed<<std::setprecision(1);
                for (auto frame: frame_records){
                    for(int i=0;i<frame.size();i++){
                        file<<frame[i]<<" ";
                        sum_val[i] += frame[i];
                        count[i] ++;
                        if(i>num_segments) num_segments=i;
                    }
                    file<<std::endl;
                }

                // 
                file<<"# Sum: ";
                for(int i=0;i<=num_segments;i++){
                    file<<sum_val[i]/count[i]<<" ";
                }
                file<<std::endl;
                file<<"# Count: ";
                for(int i=0;i<=num_segments;i++){
                    file<<count[i]<<" ";
                }

                file.close();
                std::cout<<"Write "
                        <<num_segments<<" segments and "
                        <<frame_records.size()<<" frames to "<<output_dir<<std::endl;
                return true;
            }

        private:
            int K; // number of max segments in each frame
            std::string header;
            std::vector<float> cur_frame;
            std::vector<vector<float>> frame_records;
            TicToc tic_toc;
            

    };
    
    class TimingSequence
    {
        public:
            TimingSequence(std::string header_):header(header_) {

            }

            void create_frame(int frame_id){
                cur_frame_id = frame_id;
                cur_frame.clear();           
            }
            
            void finish_frame(){
                frame_idxs.push_back(cur_frame_id);
                timings.push_back(cur_frame);
                cur_frame.clear();
            }

            void record(double duration){
                cur_frame.push_back(duration);
            }

            bool write_log(std::string output_dir)
            {
                std::ofstream file;
                file.open(output_dir);
                if (!file.is_open()){
                    std::cout<<"Failed to open file: "<<output_dir<<std::endl;
                    return false;
                }
                file<<header<<std::endl;
                file<<std::fixed<<std::setprecision(1);
                for (int i=0;i<frame_idxs.size();i++){
                    file<<frame_idxs[i]<<" ";
                    for (int j=0;j<timings[i].size();j++){
                        file<<timings[i][j]<<" ";
                    }
                    file<<std::endl;
                }
                file.close();
                std::cout<<"Write "<<frame_idxs.size()<<" frames to "<<output_dir<<std::endl;
                return true;
            
            }

        private:
            std::string header;
            int cur_frame_id;
            std::vector<float> cur_frame;

            std::vector<int> frame_idxs;
            std::vector<std::vector<float>> timings;
    };

}