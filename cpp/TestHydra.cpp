#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <DBoW2/DBoW2.h>

#include "open3d/Open3D.h"

#include "sgloop/BertBow.h"
#include "tools/IO.h"
#include "tools/TicToc.h"

typedef int NodeId;
typedef DBoW2::TemplatedVocabulary<cv::Mat, DBoW2::FORB> OrbVocabulary;
typedef DBoW2::TemplatedDatabase<cv::Mat, DBoW2::FORB> OrbDatabase;
typedef cv::Mat OrbDescriptor;
typedef std::vector<OrbDescriptor> OrbDescriptorVec;
typedef std::vector<std::pair<size_t, size_t>> KeypointMatches;
typedef std::vector<cv::DMatch> DMatchVec;
typedef fmfusion::TicToc TicToc;

cv::Ptr<cv::DescriptorMatcher> orb_feature_matcher_;

struct HydraDescriptor
{
  using Ptr = std::unique_ptr<HydraDescriptor>;
  Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> words;
  Eigen::VectorXf values;
  bool normalized = false;
  bool is_null = false;
  std::set<NodeId> nodes;
  NodeId root_node;
  Eigen::Vector3d root_position;
  //   std::chrono::nanoseconds timestamp;
};

struct CandidateDescriptor
{
  OrbDescriptor orb_desc;
  OrbDescriptorVec orb_desc_vec;
  std::vector<cv::KeyPoint> kpts;
  DBoW2::BowVector bow_vec;
};

struct TimeRecord
{
  float sg_descriptor=0.0;
  float global_match=0.0;
  float orb_feature=0.0;
  float bf_match=0.0;
  float pnp=0.0;
};

class HistgramCreator
{
public:
  HistgramCreator(const std::string &dict_file) : max_index(0)
  {
    std::ifstream file(dict_file, std::ifstream::in);
    if (!file.is_open())
    {
      std::cerr << "Error: cannot open file " << dict_file << std::endl;
      assert(false);
    }
    else
    {
      std::string line;
      while (std::getline(file, line))
      {
        std::string label;
        int index;
        index = std::stoi(line.substr(0, line.find(".")));
        label = line.substr(line.find(".") + 1, line.size() - line.find("."));

        word2int[label] = index;
        if (index > max_index)
          max_index = index;
      }
    }

    std::cout << "Histgram set to " << word2int.size() << " words and"
              << max_index << " max index." << std::endl;
  }

  void create_hist(const std::vector<std::string> &labels,
                   const std::vector<bool> &valids,
                   Eigen::VectorXf &hist, std::string &ignore_label)
  {
    hist = Eigen::VectorXf::Zero(max_index + 1);

    // for (const auto &label : labels){
    for (int i = 0; i < labels.size(); i++)
    {
      auto label = labels[i];
      if (!valids[i])
        continue;
      if (label == ignore_label)
        continue;
      if (word2int.find(label) != word2int.end())
      {
        const int idx = word2int[label];

        hist(idx) = hist(idx) + 1;
        // std::cout<<label<< hist(idx)
        //     <<"/"<<hist.sum()
        //     <<",";
      }
    }
    // std::cout<<std::endl;
  }

private:
  std::map<std::string, int> word2int;
  int max_index;
};

DEFINE_string(vocabulary_path,
              "../vocabulary/ORBvoc.yml",
              "Path to BoW vocabulary file for LoopClosureDetector module.");

std::unique_ptr<OrbVocabulary> loadOrbVocabulary()
{
  std::ifstream f_vocab(FLAGS_vocabulary_path.c_str());
  CHECK(f_vocab.good()) << "LoopClosureDetector: Incorrect vocabulary path: "
                        << FLAGS_vocabulary_path;
  f_vocab.close();
  std::cout << "Loading vocabulary from " << FLAGS_vocabulary_path << std::endl;

  auto vocab = std::make_unique<OrbVocabulary>();
  LOG(INFO) << "LoopClosureDetector:: Loading vocabulary from "
            << FLAGS_vocabulary_path;
  vocab->load(FLAGS_vocabulary_path);
  LOG(INFO) << "Loaded vocabulary with " << vocab->size() << " visual words.";
  return vocab;
}

float computeDistanceHist(const HydraDescriptor &lhs,
                          const HydraDescriptor &rhs,
                          const std::function<float(float, float)> &distance_func)
{
  assert(lhs.values.rows() == rhs.values.rows());

  float score = 0.0;
  for (int r = 0; r < lhs.values.rows(); ++r)
  {
    if (lhs.values(r, 0) == 0.0 || rhs.values(r, 0) == 0.0)
    {
      continue;
    }

    score += distance_func(lhs.values(r, 0), rhs.values(r, 0));
  }

  return score;
}

float computeDistanceBow(const HydraDescriptor &lhs,
                         const HydraDescriptor &rhs,
                         const std::function<float(float, float)> &distance_func)
{
  assert(lhs.values.rows() == lhs.words.rows());
  assert(rhs.values.rows() == rhs.words.rows());

  float score = 0.0;
  int r1 = 0;
  int r2 = 0;
  while (r1 < lhs.values.rows() && r2 < rhs.values.rows())
  {
    const uint32_t word1 = lhs.words(r1, 0);
    const uint32_t word2 = rhs.words(r2, 0);

    if (word1 == word2)
    {
      score += distance_func(lhs.values(r1, 0), rhs.values(r2, 0));
      ++r1;
      ++r2;
    }
    else if (word1 < word2)
    {
      ++r1;
    }
    else
    {
      ++r2;
    }
  }

  return score;
}

float computeDistance(const HydraDescriptor &lhs,
                      const HydraDescriptor &rhs,
                      const std::function<float(float, float)> &distance_func)
{
  if (!lhs.words.size() && !rhs.words.size())
  {
    return computeDistanceHist(lhs, rhs, distance_func);
  }
  else
  {
    return computeDistanceBow(lhs, rhs, distance_func);
  }
}

float computeCosineDistance(const HydraDescriptor &lhs, const HydraDescriptor &rhs)
{
  float lhs_scale = lhs.normalized ? 1.0f : lhs.values.norm();
  float rhs_scale = rhs.normalized ? 1.0f : rhs.values.norm();

  if (lhs_scale == 0.0f && rhs_scale == 0.0f)
  {
    return 1.0f;
  }

  float scale = lhs_scale * rhs_scale;
  // TODO(nathan) we might want a looser check than this
  if (scale == 0.0f)
  {
    scale = 1.0f; // force all zero descriptors to have 0 norm (instead of nan)
  }

  return computeDistance(
      lhs, rhs, [&scale](float lhs, float rhs)
      { return (lhs * rhs) / scale; });
}

float computeL1Distance(const HydraDescriptor &lhs, const HydraDescriptor &rhs)
{
  float lhs_scale = lhs.normalized ? 1.0f : lhs.values.lpNorm<1>();
  float rhs_scale = rhs.normalized ? 1.0f : rhs.values.lpNorm<1>();

  if (rhs_scale == 0.0f and lhs_scale == 0.0f)
  {
    return 0.0f;
  }

  lhs_scale = lhs_scale == 0.0f ? 1.0f : lhs_scale;
  rhs_scale = rhs_scale == 0.0f ? 1.0f : rhs_scale;

  const float l1_diff = computeDistance(lhs, rhs, [&](float lhs, float rhs)
                                        {
    const float lhs_val = lhs / lhs_scale;
    const float rhs_val = rhs / rhs_scale;
    return std::abs(lhs_val - rhs_val) - std::abs(lhs_val) - std::abs(rhs_val); });
  return 2.0f + l1_diff;
}

void descriptorMatToVec(const OrbDescriptor &descriptors_mat,
                        OrbDescriptorVec *descriptors_vec,
                        int L)
{
  CHECK_NOTNULL(descriptors_vec);

  // TODO(marcus): tied to ORB, need to generalize!
  // int L = orb_feature_detector_->descriptorSize();
  descriptors_vec->resize(descriptors_mat.size().height);

  for (size_t i = 0; i < descriptors_vec->size(); i++)
  {
    (*descriptors_vec)[i] =
        cv::Mat(1, L, descriptors_mat.type()); // one row only
    descriptors_mat.row(i).copyTo((*descriptors_vec)[i].row(0));
  }
}

void select_topk_frames(std::vector<std::string> &candidate_frames,
                        std::vector<float> &candidate_scores,
                        int topk = 3)
{
  std::vector<std::pair<float, std::string>> score_frame_pairs;
  assert(candidate_frames.size() == candidate_scores.size());
  if (candidate_frames.size() < topk)
  {
    topk = candidate_frames.size();
  }

  for (int i = 0; i < candidate_frames.size(); i++)
  {
    score_frame_pairs.push_back(std::make_pair(candidate_scores[i], candidate_frames[i]));
  }

  std::sort(score_frame_pairs.begin(), score_frame_pairs.end(),
            [](const std::pair<float, std::string> &a, const std::pair<float, std::string> &b)
            {
              return a.first > b.first;
            });

  candidate_frames.clear();
  candidate_scores.clear();
  for (int i = 0; i < topk; i++)
  {
    candidate_frames.push_back(score_frame_pairs[i].second);
    candidate_scores.push_back(score_frame_pairs[i].first);
    // topk_sframes.push_back(score_frame_pairs[i].second);
  }
}

int read_lcd_descriptors(const std::string &scene_dir,
                         HistgramCreator *hist_creator,
                         std::map<std::string, Eigen::VectorXf> &semantic_descriptors,
                         std::string ignore_semantics = "floor",
                         float max_distance = 5.0,
                         bool verbose = false,
                         int max_frames = 2000,
                         int frame_gap = 10)
{
  // std::map<std::string, Eigen::VectorXf> semantic_descriptors;
  std::cout << "Read LCD descriptors from " << scene_dir << std::endl;
  int latest_frame_id = -100;

  for (int id = 0; id < max_frames; id++)
  {
    std::stringstream frame_name;
    frame_name << "frame-" << std::setfill('0') << std::setw(6) << id;
    if(id-latest_frame_id<frame_gap) continue;

    Eigen::Matrix4d camera_pose;
    std::vector<Eigen::Vector3d> object_centroids;
    std::vector<std::string> object_labels;

    bool object_frame = fmfusion::IO::load_instance_info(scene_dir + "/hydra_lcd/" + frame_name.str() + ".txt",
                                                         object_centroids, object_labels);

    if (object_frame && object_centroids.size() > 0)
    {
      fmfusion::IO::load_pose_file(scene_dir + "/pose/" + frame_name.str() + ".txt", camera_pose);
      std::vector<bool> masks;
      for (auto centroid : object_centroids)
      {
        float distance = (centroid - camera_pose.block<3, 1>(0, 3)).norm();
        if (distance < max_distance)
          masks.push_back(true);
        else
          masks.push_back(false);
      }

      Eigen::VectorXf hist;
      hist_creator->create_hist(object_labels, masks, hist, ignore_semantics);
      if (hist.sum() > 0){
        semantic_descriptors[frame_name.str()] = hist;
        latest_frame_id = id;
      }
      if (verbose)
        std::cout << frame_name.str() << ": " << hist.sum() << " objects\n";
    }
  }

  return 1;
}


void computeDescriptorMatches(const OrbDescriptor& ref_descriptors,
                              const OrbDescriptor& cur_descriptors,
                              KeypointMatches* matches_match_query,
                              DMatchVec &matches,
                              double lowe_ratio = 0.7)
{
  CHECK_NOTNULL(matches_match_query);
  std::vector<DMatchVec> raw_matches;
  matches_match_query->clear();

  orb_feature_matcher_->knnMatch(cur_descriptors, ref_descriptors, raw_matches, 2u);
  if(raw_matches.empty()) return;

  const size_t& n_matches = raw_matches.size();
  for (size_t i = 0; i < n_matches; i++) {
    const DMatchVec& match = raw_matches[i];
    if (match.size() < 2) continue;
    if (match[0].distance < lowe_ratio * match[1].distance) {
      // Store trainIdx first because this represents the kpt from the
      // ref frame. For LCD, this would be the match and not the query.
      // For tracker outlier-rejection we use (ref, cur) always.
      matches_match_query->push_back(
          std::make_pair(match[0].trainIdx, match[0].queryIdx));
      cv::DMatch dmatch;
      dmatch.queryIdx = match[0].queryIdx;
      dmatch.trainIdx = match[0].trainIdx;
      dmatch.distance = match[0].distance;
      matches.push_back(dmatch);
    }
  }

}

void visualizeMatchImage(const cv::Mat& query_rgb,const cv::Mat &ref_rgb,
                         const std::vector<cv::KeyPoint>& query_keypoints,
                         const std::vector<cv::KeyPoint>& ref_keypoints,
                        const DMatchVec& match_result,
                         const std::string& output_file,
                         bool annotation=false)
{
  cv::Mat img_matches;
  // Only draw points that are in the matches
  std::vector<cv::KeyPoint> query_keypoints_match, ref_keypoints_match;
  std::vector<cv::DMatch> matches_match;
  for (const auto& match : match_result) {
    query_keypoints_match.push_back(query_keypoints[match.queryIdx]);
    ref_keypoints_match.push_back(ref_keypoints[match.trainIdx]);
    matches_match.push_back(cv::DMatch(query_keypoints_match.size() - 1,
                                       ref_keypoints_match.size() - 1, 0));
  }

  cv::drawMatches(query_rgb, query_keypoints_match, 
                  ref_rgb, ref_keypoints_match,
                  matches_match, img_matches);
                  // cv::Scalar::all(-1),
                  // cv::Scalar::all(-1),
                  // std::vector<char>(),
                  // cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  std::string text = "Matches: " + std::to_string(match_result.size());
  if(annotation)
    cv::putText(img_matches, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
  
  cv::imwrite(output_file, img_matches);
}



void recoverPose(const cv::Mat query_depth,
                 const std::vector<cv::KeyPoint>& query_keypoints,
                 const std::vector<cv::KeyPoint>& ref_keypoints,
                 const DMatchVec& matches,
                 const Eigen::Matrix3d& K,
                 Eigen::Matrix4d& pose)
{ // Solve by PnP
  const double DEPTH_SCALE = 1000.0;
  std::vector<cv::Point3f> points_3d;
  std::vector<cv::Point2f> points_2d;
  double checksum_dist = 0.0;

  for (const auto& match : matches) {
    const cv::KeyPoint& query_kpt = query_keypoints[match.queryIdx];
    const cv::KeyPoint& ref_kpt = ref_keypoints[match.trainIdx];
    const uint16_t &query_depth_val_uint = query_depth.at<uint16_t>(query_kpt.pt.y, query_kpt.pt.x);
    double query_depth_val = query_depth_val_uint / DEPTH_SCALE;
    if (query_depth_val <= 0.0) continue;
    // const cv::Point3f point_3d = cv::Point3f((query_kpt.pt.x - K(0, 2)) * query_depth_val / K(0, 0),
    //                                          (query_kpt.pt.y - K(1, 2)) * query_depth_val / K(1, 1),
    //                                          query_depth_val);

    Eigen::Vector3d point_3d = K.inverse() * Eigen::Vector3d(query_kpt.pt.x, query_kpt.pt.y, 1.0) * query_depth_val;
    cv::Point3f point_3d_cv = cv::Point3f(point_3d.x(), point_3d.y(), point_3d.z());
    points_3d.push_back(point_3d_cv);
    points_2d.push_back(ref_kpt.pt);

    checksum_dist += point_3d.norm();
  }

  if(points_3d.size() < 10) {
    pose = Eigen::Matrix4d::Identity();
    return;
  }
  
  cv::Mat rvec, tvec;
  cv::Mat K_cv = cv::Mat::eye(3, 3, CV_64F);
  K_cv.at<double>(0, 0) = K(0, 0);
  K_cv.at<double>(1, 1) = K(1, 1);
  K_cv.at<double>(0, 2) = K(0, 2);
  K_cv.at<double>(1, 2) = K(1, 2);
  
  // std::cout<<"Computing PnP with "<<points_3d.size()<<" points\n";
  // cv::solvePnP(points_3d, points_2d, K_cv, cv::Mat(), rvec, tvec);
  cv::solvePnPRansac(points_3d, points_2d, K_cv, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, cv::noArray(), cv::SOLVEPNP_EPNP);
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  pose.setIdentity();
  pose.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R.ptr<double>());
  pose.block<3, 1>(0, 3) = Eigen::Map<Eigen::Matrix<double, 3, 1>>(tvec.ptr<double>());

  // Eigen::Vector3d t_eigen;
  // Eigen::Matrix3d R_eigen;
  // cv::cv2eigen(R, R_eigen);
  // cv::cv2eigen(tvec, t_eigen);
  // pose.block<3, 3>(0, 0) = R_eigen;
  // pose.block<3, 1>(0, 3) = t_eigen;
  // pose.block<1, 4>(3, 0) << 0, 0, 0, 1;
}


int main(int argc, char *argv[])
{
  using namespace open3d;
  std::cout << "Run Hydra loop closure detection!\n";
  std::string src_scene = utility::GetProgramOptionAsString(argc, argv, "--src_scene");
  std::string ref_scene = utility::GetProgramOptionAsString(argc, argv, "--ref_scene");
  std::string output_folder = utility::GetProgramOptionAsString(argc, argv, "--output_folder");
  std::string orb_voc_dir = utility::GetProgramOptionAsString(argc, argv, "--orb_voc_dir");
  int frame_gap = utility::GetProgramOptionAsInt(argc, argv, "--frame_gap", 10);
  float distance_range = utility::GetProgramOptionAsDouble(argc, argv, "--distance_range", 5.0);
  float obj_thd = utility::GetProgramOptionAsDouble(argc, argv, "--obj_thd", 0.2);
  double bow_thd = utility::GetProgramOptionAsDouble(argc, argv, "--bow_thd", 0.2);
  double feat_thd = utility::GetProgramOptionAsDouble(argc, argv, "--feat_thd", 0.2);
  int min_feats =  utility::GetProgramOptionAsInt(argc, argv, "--min_feats", 10);
  bool verbose = utility::ProgramOptionExists(argc, argv, "--verbose");
  int max_frames = utility::GetProgramOptionAsInt(argc, argv, "--max_frames", 2000);
  int hydra_topk = utility::GetProgramOptionAsInt(argc, argv, "--hydra_topk", 10);
  int bow_topk = utility::GetProgramOptionAsInt(argc, argv, "--bow_topk", 3);
  bool annotate_viz = utility::ProgramOptionExists(argc, argv, "--annotate_viz");
  bool visualize = utility::ProgramOptionExists(argc, argv, "--visualize");

  std::string LABEL_DICT_FILE = "/home/cliuci/code_ws/OpensetFusion/torchscript/bert_bow.txt";

  // Parameters
  FLAGS_vocabulary_path = orb_voc_dir;
  Eigen::Matrix3d K;
  K << 619.18, 0.0, 336.51,
      0.0, 618.17, 246.95,
      0.0, 0.0, 1.0;

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  // Init
  HistgramCreator *hist_creator;
  std::unique_ptr<OrbDatabase> db_BoW_;
  std::unique_ptr<OrbVocabulary> vocab;
  cv::Ptr<cv::ORB> orb_feature_detector_ = cv::ORB::create(500);
  orb_feature_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_L1);

  hist_creator = new HistgramCreator(LABEL_DICT_FILE);
  vocab = loadOrbVocabulary();
  std::cout << "Loaded vocabulary\n";
  db_BoW_ = std::make_unique<OrbDatabase>(*vocab);
  std::map<std::string, CandidateDescriptor> ref_bow_map;
  std::cout << "Create DBoW2 database.\n";

  // Load and construct semantic descriptors for agent node
  std::map<std::string, Eigen::VectorXf> src_descriptors, ref_descriptors;
  read_lcd_descriptors(src_scene, hist_creator, src_descriptors, "floor", distance_range, verbose, max_frames, frame_gap);
  read_lcd_descriptors(ref_scene, hist_creator, ref_descriptors, "floor", distance_range, verbose, max_frames, frame_gap);
  std::cout << "Create " << src_descriptors.size() << " src descriptors and "
            << ref_descriptors.size() << " ref descriptors.\n";

  TimeRecord time_record;
  std::vector<double> orb_feature_count;
  int frame_count =0;
  TicToc tictoc;

  for (const auto query_frame : src_descriptors)
  {
    std::string query_frame_name = query_frame.first;
    // Eigen::VectorXf query_semantic_desc = query_frame.second;
    std::vector<cv::KeyPoint> query_keypoints;
    OrbDescriptor query_orb_descriptors;
    OrbDescriptorVec query_descriptor_vec;
    DBoW2::BowVector query_bow_vector;
    HydraDescriptor query_hydra_desc;

    int query_frame_id = std::stoi(query_frame_name.substr(query_frame_name.find("-") + 1, query_frame_name.size() - query_frame_name.find("-")));
    std::cout << "-------------- Query " << query_frame_id << " --------------\n";

    tictoc.tic();
    { // Extract query feature
      query_hydra_desc.values = query_frame.second;
      query_hydra_desc.values.normalize();
      query_hydra_desc.normalized = true;
      float t_object = tictoc.toc(true);

      cv::Mat query_rgb = cv::imread(src_scene + "/rgb/" + query_frame_name + ".png");
      if (query_rgb.empty()){
        LOG(ERROR) << "Cannot load image " << query_frame_name;
        continue;
      }
      cv::cvtColor(query_rgb, query_rgb, cv::COLOR_BGR2GRAY);

      orb_feature_detector_->detectAndCompute(query_rgb, cv::Mat(), query_keypoints, query_orb_descriptors);
      float t_orb = tictoc.toc(true);
      descriptorMatToVec(query_orb_descriptors, &query_descriptor_vec, orb_feature_detector_->descriptorSize());
      db_BoW_->getVocabulary()->transform(query_descriptor_vec, query_bow_vector);
      float t_bow = tictoc.toc(true);

      std::cout<<query_descriptor_vec.size() << " descriptors\n";
      std::cout<<"Orb feature shape "<<query_orb_descriptors.size()<<std::endl;
      std::cout<<"orb descriptor data type "<<query_orb_descriptors.type()<<std::endl;
      std::cout<<"bow vector length "<<query_bow_vector.size()<<std::endl;
      orb_feature_count.push_back(query_descriptor_vec.size());
      // (query_orb_descriptors.size().height * query_orb_descriptors.size().width);

      time_record.orb_feature += t_orb;
      time_record.sg_descriptor += t_object+t_bow;

    }

    // 1. Global search by object descriptors
    std::vector<std::string> candidate_frames;
    std::vector<float> candidate_scores;

    int count = 0;
    for (const auto ref_frame : ref_descriptors){
      std::string ref_frame_name = ref_frame.first;
      CandidateDescriptor ref_visual_descriptor;
      HydraDescriptor ref_hydra_desc;

      int ref_frame_id = std::stoi(ref_frame_name.substr(ref_frame_name.find("-") + 1, ref_frame_name.size() - ref_frame_name.find("-")));
      if (ref_frame_id > query_frame_id) continue;

      { // Extract ref feature
        ref_hydra_desc.values = ref_frame.second;
        ref_hydra_desc.values.normalize();
        ref_hydra_desc.normalized = true;

        if (ref_bow_map.find(ref_frame_name) == ref_bow_map.end())
        { // update dbow vec
          cv::Mat ref_rgb = cv::imread(ref_scene + "/rgb/" + ref_frame_name + ".png");
          if (ref_rgb.empty())
          {
            LOG(ERROR) << "Cannot load image " << ref_frame_name;
            continue;
          }
          cv::cvtColor(ref_rgb, ref_rgb, cv::COLOR_BGR2GRAY);

          orb_feature_detector_->detectAndCompute(ref_rgb, cv::Mat(), ref_visual_descriptor.kpts, ref_visual_descriptor.orb_desc);
          descriptorMatToVec(ref_visual_descriptor.orb_desc,
                             &ref_visual_descriptor.orb_desc_vec,
                             orb_feature_detector_->descriptorSize());
          db_BoW_->getVocabulary()->transform(ref_visual_descriptor.orb_desc_vec, ref_visual_descriptor.bow_vec);
          ref_bow_map[ref_frame_name] = ref_visual_descriptor;
        }
      }

      float similarity = computeCosineDistance(query_hydra_desc, ref_hydra_desc);
      if (similarity > obj_thd){
        candidate_frames.push_back(ref_frame_name);
        candidate_scores.push_back(similarity);
      }
      // std::cout<<ref_frame_name<<": "<<similarity<<",";
      count++;
    }

    if (candidate_frames.size() > 0){ // Select top-k frames
      int tmp_count = candidate_frames.size();
      select_topk_frames(candidate_frames, candidate_scores, hydra_topk);

      std::cout << "Semantic descriptor find " << candidate_frames.size()
                << "(" << tmp_count << ") frames.\n";
    }
    else {
      continue;
    }

    // 2. DBow2
    DBoW2::QueryResults query_result;
    std::vector<std::string> topk_frames;
    topk_frames.reserve(candidate_frames.size());
    for (const auto &candidate : candidate_frames){
      assert(ref_bow_map.find(candidate) != ref_bow_map.end());
      db_BoW_->add(ref_bow_map[candidate].bow_vec);
      topk_frames.emplace_back(candidate);
    }

    { // Verify DBoW results
      // float alpha_ = 0.1f;
      // float nss_factor = 1.0f;
      db_BoW_->query(query_bow_vector, query_result, bow_topk, -1);
      if (query_result.empty()){
        db_BoW_->clear();
        continue;
      }

      // Remove low scores from the QueryResults based on nss.
      DBoW2::QueryResults::iterator query_it = lower_bound(query_result.begin(),
                                                          query_result.end(),
                                                          DBoW2::Result(0, bow_thd),
                                                          DBoW2::Result::geq);

      if (query_it != query_result.end())
        query_result.resize(query_it - query_result.begin());
      
      if (query_result.empty()){// Begin grouping and checking matches.
        db_BoW_->clear();
        continue;
      }
    }
    time_record.global_match += tictoc.toc(true);
    std::cout << "DBoW2 find " << query_result.size() <<"/"<< db_BoW_->size() << " frames.\n";

    // 3. Match features
    std::string best_frame;
    DMatchVec best_dmatch_result;
    for(const DBoW2::Result &res:query_result){
      DMatchVec dmatch_result;
      KeypointMatches *query_result_matches;
      query_result_matches = new KeypointMatches;
      std::string match_frame = topk_frames[res.Id];  //[query_result[0].Id];
      computeDescriptorMatches(ref_bow_map[match_frame].orb_desc, 
                              query_orb_descriptors, 
                              query_result_matches,
                              dmatch_result,
                              feat_thd);
      if (dmatch_result.size() > best_dmatch_result.size()){
        best_frame = match_frame;
        best_dmatch_result.clear();
        best_dmatch_result = dmatch_result;
      }
    }
    time_record.bf_match += tictoc.toc(true);

    // 4. PnP
    open3d::utility::Timer timer;
    timer.Start();
    if(best_dmatch_result.size()>min_feats ){
      std::cout<<"Match "<<best_frame<<" with "<<best_dmatch_result.size()<<" kpts.\n";
      Eigen::Matrix4d pose;

      recoverPose(cv::imread(src_scene + "/depth/" + query_frame_name + ".png", cv::IMREAD_UNCHANGED),
                  query_keypoints,
                  ref_bow_map[best_frame].kpts,
                  best_dmatch_result,
                  K,
                  pose);
      fmfusion::IO::save_pose(output_folder + "/pnp_0/" + query_frame_name + "_" + best_frame+ ".txt", 
                              pose);

      // Save viz
      if (visualize){
        visualizeMatchImage(cv::imread(src_scene + "/rgb/" + query_frame_name + ".png"),
                            cv::imread(ref_scene + "/rgb/" + best_frame + ".png"),
                            query_keypoints,
                            ref_bow_map[best_frame].kpts,
                            best_dmatch_result,
                            output_folder + "/viz/" + query_frame_name + "-" + best_frame + ".png",
                            annotate_viz);
      }
    }
    time_record.pnp += tictoc.toc(true);
    timer.Stop();
    float check_pnp_Time = timer.GetDurationInMillisecond();
    std::cout<<"PnP time: "<<check_pnp_Time<<" ms\n";

    // Clear
    db_BoW_->clear();
    frame_count++;
  }

  std::cout<<"Searched between "<< src_descriptors.size()<<" and "<<ref_descriptors.size()<<" frames.\n";
  std::cout<<"Finished\n";

  fmfusion::IO::write_time({"Frames, Features, GlobalMatch, ORB, BFMatch, PnP"},
                          {frame_count, 
                          time_record.sg_descriptor,
                          time_record.global_match,
                          time_record.orb_feature,
                          time_record.bf_match,
                          time_record.pnp},
                          output_folder+"/time.txt");

  fmfusion::IO::write_time({"orb_feature_count"},
                            orb_feature_count,
                            output_folder +"/orb_count.txt");                          

  return 0;
}