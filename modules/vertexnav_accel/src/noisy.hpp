
#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <pose.hpp>
#include <types.hpp>
#include <Eigen/Dense>

#pragma once

/*
  Devoted to defining a number of the "noisy" classes: NoisyDetectionType, NoisyVertexDetection, HallucinatedVertexDetection, and NoisyVertex

*/


struct NoisyDetectionType {
  Eigen::Vector4d label;
  double prob_left_gap;
  double prob_right_gap;

  NoisyDetectionType(): prob_left_gap(0.5),
                        prob_right_gap(0.5) {
    label << 0.25, 0.25, 0.25, 0.25;
  };

  NoisyDetectionType(const char &label_in) :
      prob_left_gap(0.0), prob_right_gap(0.0) {
    Eigen::Vector4d tmp_label;
    switch (label_in) {
      case 'r':
        tmp_label << 1.0, 0.0, 0.0, 0.0;
        prob_right_gap = 1.0;
        break;
      case 'c':
        tmp_label << 0.0, 1.0, 0.0, 0.0;
        break;
      case 'l':
        tmp_label << 0.0, 0.0, 1.0, 0.0;
        prob_left_gap = 1.0;
        break;
      case 'p':
        tmp_label << 0.0, 0.0, 0.0, 1.0;
        prob_left_gap = 1.0;
        prob_right_gap = 1.0;
        break;
      default:
        throw "Value Error: char not allowed in constructor.";
    }
    label << tmp_label;
  }

  NoisyDetectionType(const Eigen::Vector4d &label_in) {
    label << label_in;
    prob_left_gap = label_in(2) + label_in(3);
    prob_right_gap = label_in(0) + label_in(3);
  }

  bool eq(const NoisyDetectionType &oth) const {
    return label == oth.label;
  }
};


class NoisyVertex {
 public:
  NoisyVertex(const Eigen::Vector2d &n_position,
              const Eigen::Matrix2d &n_cov);
  NoisyVertex(const Eigen::Vector2d &n_position,
              const Eigen::Matrix2d &n_cov,
              const Pose &n_pose);

  void set_cov(const Eigen::Matrix2d &n_cov) {
    cov = n_cov;
  }
  const Eigen::Matrix2d &get_cov() const {
    return cov;
  }

  void set_position(const Eigen::Vector2d &n_position) {
    position = n_position;
  }
  const Eigen::Vector2d &get_position() const {
    return position;
  }

  int last_updated;
  int num_detections;
  int id;

  bool is_active;
  bool is_locked;

  void add_detection(const Eigen::Vector2d &n_position,
                     const Eigen::Matrix2d &n_cov,
                     const Pose &n_pose);

  int hash() const { return id; }
  bool eq(const NoisyVertex &oth) const { return id == oth.id; }

 protected:
  static int class_id;

 private:
  Eigen::Vector2d position;
  Eigen::Matrix2d cov;
};

typedef std::shared_ptr<NoisyVertex> NoisyVertexPtr;


class NoisyVertexDetection {
public:
  double angle_rad;
  double range;
  Eigen::Vector2d position;
  Eigen::Matrix2d cov;
  Eigen::Matrix2d cov_rt;
  std::shared_ptr<NoisyDetectionType> detection_type_ptr;
  std::shared_ptr<NoisyVertex> associated_vertex_ptr;

  NoisyVertexDetection(double n_angle_rad, double n_range,
                       const std::shared_ptr<NoisyDetectionType> &n_detection_type,
                       const Eigen::Matrix2d &n_cov_rt);

  const std::shared_ptr<NoisyVertex> &get_associated_vertex() const;
  const std::shared_ptr<NoisyDetectionType> &get_detection_type() const {
    return detection_type_ptr;
  }

  void add_association(const std::shared_ptr<NoisyVertex> &new_associated_vertex) {
    associated_vertex_ptr = new_associated_vertex;
  }

  void update_props(const Pose &pose);

};

class HallucinatedVertexDetection {
 public:
  double angle_rad;
  Eigen::Vector2d position;
  int id;

  HallucinatedVertexDetection(double n_angle_rad,
                              const Eigen::Vector2d &n_position);

  HallucinatedVertexDetection(double n_angle_rad, double n_range,
                              const Pose &pose);

  const std::shared_ptr<NoisyDetectionType>& get_detection_type() const;

  void set_detection_type(const std::shared_ptr<NoisyDetectionType> &n_detection_type_ptr) {
    detection_type_ptr = n_detection_type_ptr;
  }

  int hash() const { return id; }
  bool eq(HallucinatedVertexDetection oth) const { return id == oth.id; }

 protected:
  static int class_id;
  static NoisyDetectionType default_detection_type;
  static std::shared_ptr<NoisyDetectionType> default_detection_type_ptr;
  std::shared_ptr<NoisyDetectionType> detection_type_ptr;
};

double prob_of_wall(
    const std::shared_ptr<NoisyDetectionType> &label1,
    const std::shared_ptr<NoisyDetectionType> &label2);


struct NoisyWall {
  std::vector<double> prob_list;
  std::shared_ptr<NoisyVertex> left_vertex_ptr;
  std::shared_ptr<NoisyVertex> right_vertex_ptr;
  bool is_active;

  NoisyWall(const std::shared_ptr<NoisyVertex> &n_left_vertex_ptr,
            const std::shared_ptr<NoisyVertex> &n_right_vertex_ptr):
      prob_list(), is_active(false) {
    left_vertex_ptr = n_left_vertex_ptr;
    right_vertex_ptr = n_right_vertex_ptr;
  }

  void add_detection(
      const std::shared_ptr<NoisyVertexDetection> &left_detection,
      const std::shared_ptr<NoisyVertexDetection> &right_detection,
      double false_positive_factor) {
    prob_list.push_back(false_positive_factor * prob_of_wall(
        right_detection->get_detection_type(),
        left_detection->get_detection_type()));
    update();
  }

  void add_prob_list(const std::vector<double> &oth_prob_list) {
    prob_list.insert(
        prob_list.end(),
        oth_prob_list.begin(),
        oth_prob_list.end());
    update();
  }

  int num_observations() const {
    return static_cast<int>(prob_list.size());
  }

  double prob_exists() const {
    double tot = 0;
    for (auto& p : prob_list) {
      tot += p;
    }
    return tot / (prob_list.size() + 1.0);
  }

  void update() { is_active = (prob_exists() > 0.5); }

  bool eq(const NoisyWall &oth) const {
    return (left_vertex_ptr->id == oth.left_vertex_ptr->id &&
            right_vertex_ptr->id == oth.right_vertex_ptr->id);
  }

};

typedef std::map<std::pair<int, int>, std::shared_ptr<NoisyWall>> MapPairWall;

typedef std::pair<double, Eigen::Vector2d> AnglePos;

Eigen::Vector2d eval_mat_mul(Eigen::Matrix2d in_mat, Eigen::Vector2d in_vec);

const std::map<int, NoisyVertexPtr> get_vertex_id_mapping(
    const std::vector<NoisyVertexPtr> &vertices);

const std::map<int, NoisyVertexPtr> get_vertex_remapping(
    const std::vector<NoisyVertexPtr> &vertices,
    const std::set<std::vector<int>> &topology);

std::pair<int, int> get_wall_dict_key(int id_a, int id_b);

const MapPairWall get_remapped_wall_dict(
    const MapPairWall &walls,
    const std::set<std::vector<int>> &topology,
    const std::map<int, NoisyVertexPtr> &vertex_remapping);

std::vector<Eigen::Vector2d> compute_poly_points_from_angle_pos_dat(
    const std::vector<AnglePos> &angle_pos_dat,
    const std::shared_ptr<Pose> &pose);

std::vector<Eigen::Vector2d> compute_conservative_space_hallucinated(
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<HallucinatedVertexDetection>> &observation);

std::vector<Eigen::Vector2d> compute_conservative_space_observed(
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<NoisyVertexDetection>> &observation,
    const std::map<int, NoisyVertexPtr> &vertex_remapping);
