#include "vertex_graph.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

// GTSAM
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

double dipow(double base, int exp) {
    double result = 1;
    for (;;) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }

    return result;
}

bool does_cluster_overlap(const std::vector<int> &cluster,
                          const std::set<int> &vert_ids) {
  for (const int &vid : cluster) {
    auto it = vert_ids.find(vid);
    if (it != vert_ids.end()) {
      return true;
    }
  }
  return false;
}

void ProbVertexGraph::perform_slam_update() {
  if (!DO_SLAM) { return; }

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initials;

  gtsam::noiseModel::Diagonal::shared_ptr prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(PRIOR_NOISE);
  gtsam::noiseModel::Diagonal::shared_ptr odometry_noise =
      gtsam::noiseModel::Diagonal::Sigmas(ODOMETRY_NOISE);
  gtsam::noiseModel::Diagonal::shared_ptr clustering_noise =
      gtsam::noiseModel::Diagonal::Sigmas(CLUSTERING_NOISE);

  // Get all the robot ids
  std::set<int> robot_ids;
  for (const auto &pose : poses) {
    robot_ids.insert(pose->robot_id);
  }

  // Each robot_id forms a chain of poses
  for (int rid : robot_ids) {
    // Get all poses for a particular ID
    std::vector<std::shared_ptr<Pose>> rid_poses;
    for (const auto &pose : poses) {
      if (pose->robot_id == rid) {
        rid_poses.push_back(pose);
      }
    }

    // Get all odoms for a particular ID
    std::vector<std::shared_ptr<Pose>> rid_odoms;
    for (const auto &odom : odoms) {
      if (odom->robot_id == rid) {
        rid_odoms.push_back(odom);
      }
    }

    // Add the factor for the initial pose
    const auto &initial_pose = rid_poses[0];
    gtsam::Symbol initial_pose_symbol = gtsam::Symbol('p', initial_pose->index);
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(
        initial_pose_symbol, gtsam::Pose2(
            initial_pose->x, initial_pose->y, initial_pose->yaw),
        prior_noise));
    initials.insert(initial_pose_symbol, gtsam::Pose2(
        initial_pose->x, initial_pose->y, initial_pose->yaw));

    // Add the odometry
    int num_poses = rid_poses.size();
    for (int ii = 1; ii < num_poses; ii++) {
      // Get the new symbol and add to the dictionary
      const auto &pose = rid_poses[ii];
      gtsam::Symbol old_pose_symbol = gtsam::Symbol(
          'p', rid_poses[ii-1]->index);
      gtsam::Symbol new_pose_symbol = gtsam::Symbol(
          'p', pose->index);

      // Add the odometry estimate
      const auto &odom = rid_odoms[ii-1];
      graph.add(gtsam::BetweenFactor<gtsam::Pose2>(
          old_pose_symbol, new_pose_symbol,
          gtsam::Pose2(odom->x, odom->y, odom->yaw),
          odometry_noise));

      // Add the initial estimate of pose
      initials.insert(
          new_pose_symbol, gtsam::Pose2(pose->x, pose->y, pose->yaw));
    }
  }

  // Add the observation factors
  int num_poses = poses.size();
  for (int ii = 0; ii < num_poses; ii++) {
    const auto &pose = poses[ii];
    auto pose_symbol = gtsam::Symbol('p', pose->index);

    const auto &observation = observations[ii];
    for (const auto &det : observation) {
      auto vert = det->associated_vertex_ptr;
      auto vert_symbol = gtsam::Symbol('v', vert->id);

      Eigen::Vector2d det_cov_noise;
      det_cov_noise << sqrt(det->cov_rt(1, 1)), sqrt(det->cov_rt(0, 0));
      if (!vert->is_active) {
        det_cov_noise *= 10;
      }
      auto measurement_noise = gtsam::noiseModel::Diagonal::Sigmas(
          det_cov_noise);
      graph.add(gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>(
          pose_symbol, vert_symbol,
          gtsam::Rot2(det->angle_rad), det->range,
          measurement_noise));
    }
  }

  // Add the clustering constraints
  for (const auto &cluster : topology) {
    int num_cluster_elements = cluster.size();
    if (num_cluster_elements == 1) {
      continue;
    }

    for (int ii = 0; ii < num_cluster_elements - 1; ii++) {
      auto vert_symbol_a = gtsam::Symbol('v', cluster[ii]);
      for (int jj = ii + 1; jj < num_cluster_elements; jj++) {
        auto vert_symbol_b = gtsam::Symbol('v', cluster[jj]);

        graph.add(gtsam::BetweenFactor<gtsam::Point2>(
            vert_symbol_a, vert_symbol_b,
            gtsam::Point2(0.0, 0.0),
            clustering_noise));
      }
    }
  }

  // Add the initial estimates for the vertex positions
  for (const auto &vert : vertices) {
    initials.insert(gtsam::Symbol('v', vert->id),
                    gtsam::Point2(vert->get_position()));
  }

  // Optimize
  const gtsam::LevenbergMarquardtParams params;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initials, params);
  gtsam::Values result = optimizer.optimize();

  // Set the vertex positions
  for (const auto &vert : vertices) {
    const auto &v_pos = result.at<gtsam::Point2>(
        gtsam::Symbol('v', vert->id));
    Eigen::Vector2d eigen_vert_position;
    eigen_vert_position << v_pos.x(), v_pos.y();
    vert->set_position(eigen_vert_position);
  }

  // Set the poses and update the observations
  for (int ii = 0; ii < num_poses; ii++) {
    auto pose = poses[ii];
    const auto &pose_dat = result.at<gtsam::Pose2>(
        gtsam::Symbol('p', pose->index));
    pose->x = pose_dat.x();
    pose->y = pose_dat.y();
    pose->yaw = pose_dat.theta();

    const auto &observation = observations[ii];
    for (const auto &det : observation) {
      det->update_props(*pose);
    }
  }
}

std::vector<std::shared_ptr<HallucinatedVertexDetection>> compute_hypothetical_observation(
    const std::shared_ptr<ProposedWorldCore> &world,
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<NoisyVertexDetection>> &observation) {
  auto poly_points = compute_conservative_space_observed(
      pose, observation, world->vertex_remapping);
  auto h_vertex_positions = world->get_vertices_for_pose(
      pose, poly_points);

  std::vector<std::shared_ptr<HallucinatedVertexDetection>> h_observation;
  for (const Eigen::Vector2d &position : h_vertex_positions) {
    double angle_rad = atan2(position(1) - pose->y,
                             position(0) - pose->x);
    h_observation.push_back(std::make_shared<HallucinatedVertexDetection>(
        angle_rad, position));
  }
  std::sort(h_observation.begin(), h_observation.end(),
       [] (const std::shared_ptr<HallucinatedVertexDetection> h_det_a,
           const std::shared_ptr<HallucinatedVertexDetection> h_det_b) -> bool {
         return h_det_a->angle_rad < h_det_b->angle_rad;
       });

  return h_observation;
}

typedef std::pair<int, int> PosKey;
typedef std::shared_ptr<HallucinatedVertexDetection> HVDptr;

inline PosKey get_position_key(const Eigen::Vector2d &position) {
  return std::make_pair(static_cast<int>(position(0) * 1000000),
                        static_cast<int>(position(1) * 1000000));
}

double compute_observation_log_likelihood(
    const std::shared_ptr<ProposedWorldCore> &world,
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<NoisyVertexDetection>> observation,
    const std::map<PosKey, NoisyVertexPtr> vertex_positions_memo,
    double log_fp_rate, double log_fn_rate) {
  double log_prob = 0.0;
  int match_count = 0;
  int fp_count = 0;

  std::vector<HVDptr> h_observation
      = compute_hypothetical_observation(world, pose, observation);

  std::map<int, int> h_det_match_count_map;  // h_det->id => num_matches
  std::map<int, double> h_det_dist_prob_map;  // h_det->id => dist_prob
  std::map<int, NoisyVertexPtr> h_det_vert_map;  // h_det->id => parent vertex
  std::map<PosKey, HVDptr> h_det_map;

  for (const auto &h_det : h_observation) {
    PosKey h_pos_key = get_position_key(h_det->position);
    h_det_map[h_pos_key] = h_det;
    h_det_match_count_map[h_det->id] = 0;
    h_det_vert_map[h_det->id] = vertex_positions_memo.find(h_pos_key)->second;
  }

  for (const auto &r_det : observation) {
    PosKey r_pos_key = get_position_key(world->vertex_remapping[
        r_det->associated_vertex_ptr->id]->get_position());

    const auto &it = h_det_map.find(r_pos_key);
    if (it != h_det_map.end()) {
      // h_det found
      auto h_det = it->second;
      Eigen::Vector2d dx = r_det->position - h_det->position;
      double dist_prob = -(r_det->cov.inverse() * dx).dot(dx);

      int count = h_det_match_count_map[h_det->id]++;
      if (count > 0) {
        // We've already seen this h_det
        // Add a false positive, avg the det types, get max dist prob
        fp_count++;
        Eigen::Vector4d o_det_type, n_det_type;
        o_det_type << h_det->get_detection_type()->label;
        n_det_type << r_det->get_detection_type()->label;
        h_det->set_detection_type(std::make_shared<NoisyDetectionType>(
            (count * o_det_type + n_det_type) / (1.0 + count)));
        h_det_dist_prob_map[h_det->id] = max(
            h_det_dist_prob_map[h_det->id], dist_prob);
      } else {
        // First time we're seeing this h_det
        h_det->set_detection_type(r_det->get_detection_type());
        h_det_dist_prob_map[h_det->id] = dist_prob;
        match_count++;
      }
    } else {
      // h_det not found (incrememnt num false positives)
      fp_count++;
    }
  }  // end for (r_det)

  // Update the probability using the matches/misses
  log_prob += (h_observation.size() - match_count) * log_fn_rate;
  log_prob += (fp_count) * log_fp_rate;

  // Add the distance likelihoods
  for (const auto &it : h_det_dist_prob_map) {
    log_prob += max(log_fp_rate, it.second);
  }

  // Wall likelihood computation
  double wall_prob = 1.0;
  int num_h_dets = h_observation.size();
  for (int rind = 0; rind < num_h_dets; ++rind) {
    // Some helper variables
    int lind = (rind + 1) % num_h_dets;
    auto det_L = h_observation[lind];
    auto vert_L = h_det_vert_map[det_L->id];
    auto det_R = h_observation[rind];
    auto vert_R = h_det_vert_map[det_R->id];
    double high_angle = det_L->angle_rad;
    double low_angle = det_R->angle_rad;

    if (high_angle < low_angle) {
      high_angle += 2 * M_PI;
    }
    if (high_angle - low_angle >= M_PI) {
      continue;
    }

    double wall_exists_prob = 0.0;
    auto wall_key = get_wall_dict_key(vert_L->id, vert_R->id);
    const auto &it = world->remapped_wall_dict.find(wall_key);
    if (it != world->remapped_wall_dict.end()) {
      // wall found
      wall_exists_prob = it->second->is_active;
    }

    double obs_wall_prob = prob_of_wall(
        det_R->get_detection_type(), det_L->get_detection_type());

    wall_prob *= max(1 - abs(obs_wall_prob - wall_exists_prob), 0.25);
  }  // end for (walls)

  log_prob += log(wall_prob);

  return log_prob;
}


double ProbVertexGraph::compute_world_log_prob(
    const std::shared_ptr<ProposedWorldCore> &world,
    const std::vector<PoseObsPair> &pose_obs_pairs) const {
  std::map<PosKey, NoisyVertexPtr> vertex_positions_memo;
  for (const auto &it : world->vertex_remapping) {
    vertex_positions_memo.insert(std::make_pair(
        get_position_key(it.second->get_position()), it.second));
  }

  double log_fp_rate = log(false_positive_rate);
  double log_fn_rate = log(false_negative_rate);

  double prob = 2 * world->vertices.size() * log(false_positive_rate);
  for (const auto &pose_obs_pair : pose_obs_pairs) {
    prob += compute_observation_log_likelihood(
        world, pose_obs_pair.first, pose_obs_pair.second,
        vertex_positions_memo,
        log_fp_rate, log_fn_rate);
  }

  return prob;
}


double ProbVertexGraph::compute_world_log_prob_full(
    const std::shared_ptr<ProposedWorldCore> &world) const {
  std::map<PosKey, NoisyVertexPtr> vertex_positions_memo;
  for (const auto &it : world->vertex_remapping) {
    vertex_positions_memo.insert(std::make_pair(
        get_position_key(it.second->get_position()), it.second));
  }

  double log_fp_rate = log(false_positive_rate);
  double log_fn_rate = log(false_negative_rate);

  double prob = 2 * world->vertices.size() * log(false_positive_rate);
  int num_poses = poses.size();
  for (int ii = 0; ii < num_poses; ii++) {
    prob += compute_observation_log_likelihood(
        world, poses[ii], observations[ii],
        vertex_positions_memo,
        log_fp_rate, log_fn_rate);
  }

  return prob;
}
