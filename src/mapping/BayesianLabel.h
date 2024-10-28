#ifndef BAYESIANLABEL_H
#define BAYESIANLABEL_H

#include <vector>
#include <string>
#include <iostream>
#include <eigen3/Eigen/Core>

#include "Detection.h"
#include "tools/Utility.h"

namespace fmfusion
{

class BayesianLabel
{
    public:
        /// \brief  Load the likelihood matrix from file.
        BayesianLabel(const std::string &likelihood_file, bool verbose=false)
        {
            std::cout<<"Semantic fusion load likelihood matrix from "
                    <<likelihood_file<<std::endl;
            if(load_likelihood_matrix(likelihood_file, verbose)){
                is_loaded = true;
            }
            else{
                std::cerr<<"Failed to load likelihood matrix from "
                         <<likelihood_file<<std::endl;
            }

        };
        ~BayesianLabel(){};

        /// \brief  Update the probability vector from the measurements.
        bool update_measurements(const std::vector<LabelScore> &measurements,
                                Eigen::VectorXf &probability_vector)
        {
            if(measurements.empty() || !is_loaded) return false;

            int rows = measurements.size();
            int cols = predict_label_vec.size();
            probability_vector = Eigen::VectorXf::Zero(cols);
            for(int i=0;i<rows;i++){
                std::string measure_label = measurements[i].first;
                float measure_score = measurements[i].second;
                if(measure_label_map.find(measure_label)!=measure_label_map.end()){
                    probability_vector += measure_score 
                                        * likelihood_matrix.row(measure_label_map[measure_label]).transpose();
                }
            }

            return true;
        }
        
        bool update_measurements(const std::unordered_map<std::string, float> &measurements,
                                Eigen::VectorXf &probability_vector)
        {
            if(measurements.empty() || !is_loaded) return false;
            std::vector<LabelScore> measurements_vec;
            for(const auto &label_score:measurements){
                measurements_vec.emplace_back(std::make_pair(label_score.first, label_score.second));
            }

            return update_measurements(measurements_vec, probability_vector);
        }

        std::vector<std::string> get_label_vec()const{
            return predict_label_vec;
        }

        int get_num_classes() const{
            return predict_label_vec.size();
        }

    
    private:

        /// \brief  Load the likelihood matrix from text file.
        bool load_likelihood_matrix(const std::string &likelihood_file, bool verbose)
        {
            std::ifstream file(likelihood_file);
            std::stringstream msg;
            if(file.is_open()){
                // read in lines
                std::string line;
                std::getline(file, line); // header
                std::getline(file, line); // 
                // std::cout<<line<<std::endl;
                measure_label_vec = utility::split_str(line, ",");

                std::getline(file, line); //
                predict_label_vec = utility::split_str(line, ",");
                
                int rows = measure_label_vec.size();
                int cols = predict_label_vec.size();
                std::cout<<"rows: "<<rows<<" cols: "<<cols<<std::endl;
                likelihood_matrix = Eigen::MatrixXf::Zero(rows, cols);
                for(int i=0;i<rows;i++){
                    std::getline(file, line);
                    std::vector<std::string> values = utility::split_str(line, ",");
                    for(int j=0;j<cols;j++){
                        likelihood_matrix(i,j) = std::stof(values[j]);
                    }
                }
                
                msg<<"Measured label-set: ";
                for (int i=0;i<measure_label_vec.size();i++) {
                    measure_label_map[measure_label_vec[i]] = i;
                    msg<<measure_label_vec[i]<<" ";
                }
                msg<<"\n"
                   <<"Predict label-set: ";
                for (auto label : predict_label_vec) {
                    msg<<label<<" ";
                }
                msg<<"\n"
                   <<"Likelihood matrix: "<<likelihood_matrix.rows()<<"x"
                   <<likelihood_matrix.cols()<<std::endl;

                assert(measure_label_vec.size()==likelihood_matrix.rows());
                assert(predict_label_vec.size()==likelihood_matrix.cols());
                if(verbose) std::cout<<msg.str();
                return true;
            }
            else return false;

        }

    private:
        bool is_loaded = false;
        std::vector<std::string> predict_label_vec;
        std::vector<std::string> measure_label_vec;
        std::map<std::string, int> measure_label_map;
        
        Eigen::MatrixXf likelihood_matrix;


};

}

#endif // BAYESIANLABEL_H
