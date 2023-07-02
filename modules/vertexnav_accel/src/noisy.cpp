#include <noisy.hpp>

#include <utility>
#include <set>
#include <vector>


NoisyVertex::NoisyVertex(const Eigen::Vector2d &n_position,
                         const Eigen::Matrix2d &n_cov):
    last_updated(-1), num_detections(0),
    is_active(true), is_locked(false),
    position(n_position), cov(n_cov) {
  id = class_id++;
}


NoisyVertex::NoisyVertex(const Eigen::Vector2d &n_position,
                         const Eigen::Matrix2d &n_cov,
                         const Pose &n_pose):
    last_updated(n_pose.index), num_detections(1),
    is_active(true), is_locked(false),
    position(n_position), cov(n_cov) {
  id = class_id++;
}

void NoisyVertex::add_detection(const Eigen::Vector2d &n_position,
                                const Eigen::Matrix2d &n_cov,
                                const Pose &n_pose) {
  // Compute Kalman gain (K)
  Eigen::Matrix2d K = cov *(cov + n_cov).inverse();

  // Perform the Kalman update
  position << position + K * (n_position - position);
  cov << (Eigen::Matrix2d::Identity() - K) * cov;

  // Some additional parameter updates.
  num_detections += 1;
  last_updated = n_pose.index;
}

int NoisyVertex::class_id = 0;


NoisyVertexDetection::NoisyVertexDetection(
    double n_angle_rad, double n_range,
    const std::shared_ptr<NoisyDetectionType> &n_detection_type_ptr,
    const Eigen::Matrix2d &n_cov_rt):
      angle_rad(n_angle_rad),
      range(n_range),
      cov_rt(n_cov_rt) {
  detection_type_ptr = n_detection_type_ptr;
}

const std::shared_ptr<NoisyVertex> &NoisyVertexDetection::get_associated_vertex() const {
  if (associated_vertex_ptr) {
    return associated_vertex_ptr;
  } else {
    throw "ValueError: This detection does not have an associated vertex";
  }
}

void NoisyVertexDetection::update_props(const Pose &pose) {
  double world_theta = angle_rad + pose.yaw;
  double cov_r = cov_rt(0, 0);
  double r_sq_cov_theta = range * range * cov_rt(1, 1);
  double sin_th = sin(world_theta);
  double cos_th = cos(world_theta);
  double sin_th_sq = sin_th * sin_th;
  double cos_th_sq = cos_th * cos_th;
  double Q11 = r_sq_cov_theta * sin_th_sq + cov_r * cos_th_sq;
  double Q22 = r_sq_cov_theta * cos_th_sq + cov_r * sin_th_sq;
  double Q12 = (cov_r - r_sq_cov_theta) * cos_th * sin_th;

  position << pose.x + range * cos_th, pose.y + range * sin_th;
  cov << Q11, Q12, Q12, Q22;
}


// HallucinatedVertexDetection
int HallucinatedVertexDetection::class_id = 0;
void null_deleter(NoisyDetectionType *) {}
NoisyDetectionType HallucinatedVertexDetection::default_detection_type = NoisyDetectionType();
std::shared_ptr<NoisyDetectionType> HallucinatedVertexDetection::default_detection_type_ptr = std::shared_ptr<NoisyDetectionType>(&HallucinatedVertexDetection::default_detection_type, &null_deleter);

HallucinatedVertexDetection::HallucinatedVertexDetection(
    double n_angle_rad, double n_range, const Pose &pose) :
      angle_rad(n_angle_rad) {
  id = ++class_id;

  Eigen::Vector2d tmp_position;
  tmp_position << pose.x + n_range * cos(n_angle_rad),
      pose.y + n_range * sin(n_angle_rad);
  position = tmp_position;
}

HallucinatedVertexDetection::HallucinatedVertexDetection(
    double n_angle_rad, const Eigen::Vector2d &n_position) :
      angle_rad(n_angle_rad),
      position(n_position) {
  id = ++class_id;
}

const std::shared_ptr<NoisyDetectionType>& HallucinatedVertexDetection::get_detection_type() const {
  if (detection_type_ptr) {
    return detection_type_ptr;
  } else {
    return default_detection_type_ptr;
  }
}



// NoisyWall


Eigen::Vector2d eval_mat_mul(Eigen::Matrix2d in_mat, Eigen::Vector2d in_vec) {
  return in_mat * in_vec;
}

double prob_of_wall(
    const std::shared_ptr<NoisyDetectionType> &label1,
    const std::shared_ptr<NoisyDetectionType> &label2) {
  return (1 - 0.5 * (label1->prob_left_gap + label2->prob_right_gap));
}


const std::map<int, NoisyVertexPtr> get_vertex_id_mapping(
    const std::vector<NoisyVertexPtr> &vertices) {
  std::map<int, NoisyVertexPtr> vertex_id_mapping;
  for (auto &vert : vertices) {
    vertex_id_mapping.insert(std::make_pair(vert->id, vert));
  }
  return vertex_id_mapping;
}


const std::map<int, NoisyVertexPtr> get_vertex_remapping(
    const std::vector<NoisyVertexPtr> &vertices,
    const std::set<std::vector<int>> &topology) {
  std::map<int, NoisyVertexPtr> vertex_id_mapping =
      get_vertex_id_mapping(vertices);
  std::map<int, NoisyVertexPtr> vertex_remapping;

  for (const auto &cluster : topology) {
    if (cluster.size() == 1) {
      vertex_remapping.insert(std::make_pair(
          cluster[0], vertex_id_mapping[cluster[0]]));
    } else {
      Eigen::Matrix2d sum_cov_inv = Eigen::Matrix2d::Zero();
      Eigen::Vector2d sum_weighted_means = Eigen::Vector2d::Zero();
      for (const auto &vid : cluster) {
        auto vert = vertex_id_mapping[vid];
        auto cov_inv = (vert->get_cov()).inverse();
        sum_cov_inv += cov_inv;
        sum_weighted_means += cov_inv * vert->get_position();
      }

      auto cov = sum_cov_inv.inverse();
      NoisyVertexPtr cluster_vert = std::make_shared<NoisyVertex>(
          cov * sum_weighted_means, cov);

      for (const auto &vid : cluster) {
        vertex_remapping.insert(std::make_pair(vid, cluster_vert));
      }
    }
  }

  return vertex_remapping;
}

std::pair<int, int> get_wall_dict_key(int id_a, int id_b) {
  if (id_a < id_b) {
    return std::pair<int, int>(id_a, id_b);
  } else {
    return std::pair<int, int>(id_b, id_a);
  }
}

const MapPairWall get_remapped_wall_dict(
    const MapPairWall &walls,
    const std::set<std::vector<int>> &topology,
    const std::map<int, NoisyVertexPtr> &vertex_remapping) {
  // Get list of unmapped vert ids
  std::set<int> unremapped_verts;
  for (const auto &cluster : topology) {
    if (cluster.size() == 1) {
      unremapped_verts.insert(cluster[0]);
    }
  }

  MapPairWall walls_remapped;
  for (auto pair_wall : walls) {
    auto va = vertex_remapping.find(pair_wall.first.first)->second;
    auto vb = vertex_remapping.find(pair_wall.first.second)->second;
    auto remapped_key = get_wall_dict_key(va->id, vb->id);

    // If neither vertex is remapped, use the existing wall.
    if (unremapped_verts.find(va->id) != unremapped_verts.end() &&
        unremapped_verts.find(vb->id) != unremapped_verts.end()) {
      walls_remapped.insert(std::make_pair(remapped_key, pair_wall.second));
      continue;
    }

    const auto &it = walls_remapped.find(remapped_key);
    std::shared_ptr<NoisyWall> remapped_wall;
    if (it != walls_remapped.end()) {
      // Get the wall from the map
      remapped_wall = it->second;
    } else {
      // Create a new wall and add it to the list
      remapped_wall = std::make_shared<NoisyWall>(va, vb);
      walls_remapped.insert(std::make_pair(
          remapped_key, remapped_wall));
    }
    remapped_wall->add_prob_list(pair_wall.second->prob_list);
  }

  return walls_remapped;
}

std::vector<Eigen::Vector2d> compute_poly_points_from_angle_pos_dat(
    const std::vector<AnglePos> &angle_pos_dat,
    const std::shared_ptr<Pose> &pose) {
  std::vector<Eigen::Vector2d> poly_points;
  int num_dats = angle_pos_dat.size();
  for (int rind = 0; rind < num_dats; ++rind) {
    // Some helper variables
    int lind = (rind + 1) % num_dats;
    double low_angle = angle_pos_dat[rind].first;
    double high_angle = angle_pos_dat[lind].first;

    // Add the current point
    poly_points.push_back(angle_pos_dat[rind].second);

    // If necessary, add the robot pose
    if (high_angle < low_angle) {
      high_angle += 2 * M_PI;
    }
    if ((high_angle - low_angle) >= M_PI) {
      Eigen::Vector2d pose_position;
      pose_position << pose->x, pose->y;
      poly_points.push_back(pose_position);
    }
  }

  // By convention, the final point is a duplicate of the first
  if (num_dats > 0) {
    poly_points.push_back(angle_pos_dat[0].second);
  }

  return poly_points;
}

std::vector<Eigen::Vector2d> compute_conservative_space_hallucinated(
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<HallucinatedVertexDetection>> &observation) {
  std::vector<AnglePos> angle_pos_dat;
  for (auto& h_det : observation) {
    angle_pos_dat.push_back(std::make_pair(
        h_det->angle_rad, h_det->position));
  }

  std::sort(angle_pos_dat.begin(), angle_pos_dat.end(),
            [](const AnglePos &a, const AnglePos &b) -> bool {
              return a.first < b.first;});

  return compute_poly_points_from_angle_pos_dat(angle_pos_dat, pose);
}


std::vector<Eigen::Vector2d> compute_conservative_space_observed(
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<NoisyVertexDetection>> &observation,
    const std::map<int, NoisyVertexPtr> &vertex_remapping) {
  std::vector<AnglePos> angle_pos_dat;
  for (auto& det : observation) {
    Eigen::Vector2d position = vertex_remapping.find(
        det->associated_vertex_ptr->id)->second->get_position();
    float angle_rad = atan2(position[1] - pose->y,
                            position[0] - pose->x);
    angle_pos_dat.push_back(std::make_pair(
        angle_rad, position));
  }

  std::sort(angle_pos_dat.begin(), angle_pos_dat.end(),
            [](const AnglePos &a, const AnglePos &b) -> bool {
              return a.first < b.first; });

  return compute_poly_points_from_angle_pos_dat(angle_pos_dat, pose);
}
