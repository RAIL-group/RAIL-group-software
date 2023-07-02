#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <Hungarian/Hungarian.h>
#include <noisy.hpp>
#include <cgal_ex.hpp>

typedef std::vector<std::shared_ptr<NoisyVertexDetection>> RealObservation;
typedef std::pair<std::set<std::vector<int>>, std::set<std::vector<int>>> TopologyOperation;
typedef std::pair<std::shared_ptr<Pose>, RealObservation> PoseObsPair;

double dipow(double base, int exp);

bool does_cluster_overlap(const std::vector<int> &cluster,
                          const std::set<int> &vert_ids);

struct NoisyVertexInfo {
  bool is_active;
  bool is_locked;
  Eigen::Vector2d position;

  NoisyVertexInfo(const NoisyVertexPtr &vert) :
      is_active(vert->is_active),
      is_locked(vert->is_locked),
      position(vert->get_position()) {}
};

struct SamplingState {
  std::set<std::vector<int>> topology;
  double log_prob;
  std::map<int, std::shared_ptr<NoisyVertexInfo>> vertex_info_map;
  std::map<int, std::array<double, 3>> pose_info_map;

  SamplingState(const std::vector<NoisyVertexPtr> &vertices,
                const std::vector<std::vector<int>> &n_topology,
                double n_log_prob,
                const std::vector<std::shared_ptr<Pose>> &poses) :
      log_prob(n_log_prob) {
    for (const auto &cluster : n_topology) {
      topology.insert(cluster);
    }
    for (const auto &vert : vertices) {
      vertex_info_map[vert->id] =
          std::make_shared<NoisyVertexInfo>(vert);
    }
    for (const auto &pose : poses) {
      pose_info_map[pose->index] = std::array<double, 3>(
          {pose->x, pose->y, pose->yaw});
    }
  }

  SamplingState(const std::vector<NoisyVertexPtr> &vertices,
                const std::set<std::vector<int>> &n_topology,
                double n_log_prob,
                const std::vector<std::shared_ptr<Pose>> &poses) :
      topology(n_topology), log_prob(n_log_prob) {
    for (const auto &vert : vertices) {
      vertex_info_map[vert->id] =
          std::make_shared<NoisyVertexInfo>(vert);
    }
    for (const auto &pose : poses) {
      pose_info_map[pose->index] = std::array<double, 3>(
          {pose->x, pose->y, pose->yaw});
    }
  }

  std::vector<std::vector<int>> get_topology_py() const {
    std::vector<std::vector<int>> top_out(topology.begin(), topology.end());
    return top_out;
  }
};


class ProposedWorldCore : public FastWorld {
 public:
  std::set<std::vector<int>> topology;
  std::map<int, NoisyVertexPtr> vertex_remapping;
  MapPairWall remapped_wall_dict;

  ProposedWorldCore(const std::vector<std::pair<NoisyVertexPtr, NoisyVertexPtr>> &segs,
                    const std::vector<NoisyVertexPtr> &verts,
                    const std::set<std::vector<int>> &n_topology,
                    const std::map<int, NoisyVertexPtr> &n_vertex_remapping,
                    const MapPairWall &n_remapped_wall_dict) :
      topology(n_topology),
      vertex_remapping(n_vertex_remapping),
      remapped_wall_dict(n_remapped_wall_dict) {

    for (const auto &vert : verts) {
      vertices.push_back(vert->get_position());
    }

    std::vector<Segment_2> segments;
    for (const auto &seg : segs) {
      segments.push_back(Segment_2(
          Point_2(seg.first->get_position()(0), seg.first->get_position()(1)),
          Point_2(seg.second->get_position()(0), seg.second->get_position()(1))));
    }

    // Needed so that the ray casting will never go "out of bounds"
    segments.push_back(Segment_2(
        Point_2(-500.1, -500.2), Point_2(-500.3, 500.4)));
    segments.push_back(Segment_2(
        Point_2(-500.3, 500.4), Point_2(500.5, 500.6)));
    segments.push_back(Segment_2(
        Point_2(500.5, 500.6), Point_2(500.7, -500.8)));
    segments.push_back(Segment_2(
        Point_2(500.7, -500.8), Point_2(-500.1, -500.2)));

    // insert geometry into the arrangement
    CGAL::insert(env, segments.begin(), segments.end());
    tev = std::make_shared<TEV>(env);
  }

  std::vector<std::vector<int>> get_topology_py() const {
    std::vector<std::vector<int>> top_out(topology.begin(), topology.end());
    return top_out;
  }

  VEV get_vertices_for_pose(const std::shared_ptr<Pose> &pose,
                            const VEV &poly_bound) const {
    return getVisPolyBounded(pose->x, pose->y, poly_bound);
  }

};


std::vector<std::shared_ptr<HallucinatedVertexDetection>> compute_hypothetical_observation(
    const std::shared_ptr<ProposedWorldCore> &world,
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<NoisyVertexDetection>> &observation);


class ProbVertexGraph {
 public:
  std::vector<std::shared_ptr<NoisyVertex>> vertices;
  std::set<std::vector<int>> topology;
  MapPairWall walls;

  std::vector<std::shared_ptr<Pose>> poses;
  std::vector<std::shared_ptr<Pose>> odoms;
  std::vector<RealObservation> observations;

  double false_positive_rate = 0.200;
  double false_negative_rate = 0.200;

  std::vector<TopologyOperation> disallowed_topology_operations;
  double TOPOLOGY_LOCK_LOG_THRESH = 1000;

  bool DO_SLAM = true;
  bool DO_SAMPLE_TOPOLOGY = true;

  Eigen::Vector3d PRIOR_NOISE;
  Eigen::Vector3d ODOMETRY_NOISE;
  Eigen::Vector2d CLUSTERING_NOISE;

  ProbVertexGraph() {}

  std::vector<std::vector<int>> get_topology_py() const {
    std::vector<std::vector<int>> top_out(topology.begin(), topology.end());
    return top_out;
  }

  void set_topology_py(const std::vector<std::vector<int>> &top_in) {
    topology.clear();
    for (auto &it : top_in) {
      topology.insert(it);
    }
  }

  void add_observation_pose(RealObservation n_observation,
                            std::shared_ptr<Pose> n_pose,
                            int association_window = -1) {
    for (const auto &det: n_observation) {
      det->update_props(*n_pose);
    }

    associate_detections(n_observation, n_pose,
                         association_window);

    poses.push_back(n_pose);
    observations.push_back(n_observation);

    // Add the wall observations
    int num_dets = n_observation.size();
    for (int rind = 0; rind < num_dets; ++rind) {
      auto det_R = n_observation[rind];
      for (int shift = 1; shift < num_dets; ++shift) {
        int lind = (rind + shift) % num_dets;
        auto det_L = n_observation[lind];
        add_wall_detection(det_L, det_R, shift-1);
      }
    }
  }

  void add_observation_odom(RealObservation n_observation,
                            std::shared_ptr<Pose> n_odom,
                            int association_window = -1) {
    // Compute odometry via the previous pose to match robot_id's
    int pose_rid_int = -1;
    for (int ii = 0; ii < poses.size(); ++ii) {
      if (poses[ii]->robot_id == n_odom->robot_id) {
        pose_rid_int = ii;
      }
    }
    std::shared_ptr<Pose> n_pose = std::make_shared<Pose>(
        (*n_odom) * (*poses[pose_rid_int]));
    odoms.push_back(n_odom);
    add_observation_pose(n_observation, n_pose, association_window);
  }

  void add_wall_detection(
      const std::shared_ptr<NoisyVertexDetection> &det_L,
      const std::shared_ptr<NoisyVertexDetection> &det_R,
      int num_false_positives) {
    double dtheta = fmod(
        det_L->angle_rad - det_R->angle_rad + 2*M_PI, 2*M_PI);

    if (dtheta >= M_PI) {
      return;
    }

    auto vert_L = det_L->get_associated_vertex();
    auto vert_R = det_R->get_associated_vertex();

    std::pair<int, int> key = get_wall_dict_key(vert_L->id, vert_R->id);
    auto it = walls.find(key);
    std::shared_ptr<NoisyWall> wall;
    if (it != walls.end()) {
      // Get the wall from the map
      wall = it->second;
    } else {
      // Create a new wall and add it to the list
      wall = std::make_shared<NoisyWall>(vert_L, vert_R);
      walls[key] = wall;
    }

    double fp_factor = dipow(false_positive_rate,
                             num_false_positives);
    wall->add_detection(det_L, det_R, fp_factor);
  }

  void associate_detections(const RealObservation &n_observation,
                            const std::shared_ptr<Pose> &n_pose,
                            int association_window) {
    std::vector<std::shared_ptr<NoisyVertex>> all_vertices;
    const int num_observations(observations.size());
    if (association_window <= 0) {
      all_vertices = vertices;
    } else {  // Limit by robot_id
      // Get all observations for a particular ID
      std::vector<RealObservation> rid_observations;
      for (int ii = 0; ii < poses.size(); ++ii) {
        if (poses[ii]->robot_id == n_pose->robot_id) {
          rid_observations.push_back(observations[ii]);
        }
      }

      // Now find the vertices associated with those observations
      int num_rid_observations = rid_observations.size();
      std::set<int> vert_ids;
      for (int ii = std::max(num_rid_observations - association_window, 0);
           ii < num_rid_observations;
           ii++) {
        for (const auto &det: rid_observations[ii]) {
          vert_ids.insert(det->associated_vertex_ptr->id);
        }
      }

      for (auto &vert : vertices) {
        if (vert_ids.find(vert->id) != vert_ids.end()) {
          all_vertices.push_back(vert);
        }
      }
    }

    const int num_verts(all_vertices.size());
    const int num_dets(n_observation.size());

    // Build the cost matrix
    std::vector<std::vector<double>> cost_mat;
    for (int dind = 0; dind < num_dets; ++dind) {
      auto det = n_observation[dind];
      Eigen::Matrix2d det_inv_cov = (det->cov).inverse();
      std::vector<double> cost_mat_row(num_dets + num_verts, 1.0);
      for (int vind = 0; vind < num_verts; ++vind) {
        auto vert = all_vertices[vind];
        Eigen::Vector2d dpos = vert->get_position() - det->position;
        cost_mat_row[vind] = sqrt( dpos.dot( det_inv_cov * dpos ));
      }
      cost_mat.push_back(cost_mat_row);
    }
    vector<int> assignment;

    HungarianAlgorithm HungAlgo;
    double cost = HungAlgo.Solve(cost_mat, assignment);

    for (int dind = 0; dind < num_dets; ++dind) {
      int vind = assignment[dind];
      const auto &det = n_observation[dind];

      if (vind >= num_verts) {
        std::shared_ptr<NoisyVertex> new_vert = std::make_shared<NoisyVertex>(
            det->position, det->cov, *n_pose);
        det->add_association(new_vert);
        vertices.push_back(new_vert);
        topology.insert({new_vert->id});
      } else {
        auto vert = all_vertices[vind];
        vert->add_detection(
            det->position, det->cov, *n_pose);
        det->add_association(vert);
      }
    }

  }

  std::shared_ptr<ProposedWorldCore> get_proposed_world_py(
      const std::vector<std::vector<int>> &world_topology_vec) {
    std::set<std::vector<int>> world_topology_set;
    for (auto &it : world_topology_vec) {
      world_topology_set.insert(it);
    }
    return get_proposed_world(world_topology_set);
  }

  std::shared_ptr<ProposedWorldCore> get_proposed_world(
      const std::set<std::vector<int>> &world_topology) {
    // Get the active verts
    std::set<int> active_vert_ids;
    for (const auto &vert : vertices) {
      if (vert->is_active && vert->num_detections > 2) {
        active_vert_ids.insert(vert->id);
      }
    }

    // Get the "active clusters"
    std::set<std::vector<int>> active_clusters;
    for (const auto &cluster : world_topology) {
      if (does_cluster_overlap(cluster, active_vert_ids)) {
        active_clusters.insert(cluster);
      }
    }

    // Build the remapping objects
    auto vertex_remapping = get_vertex_remapping(vertices, world_topology);
    auto remapped_wall_dict = get_remapped_wall_dict(
        walls, active_clusters, vertex_remapping);

    // Get the world data
    std::vector<NoisyVertexPtr> cluster_verts;
    std::set<int> cluster_vert_ids;
    std::vector<std::pair<NoisyVertexPtr, NoisyVertexPtr>> wall_verts;
    for (const std::vector<int> &cluster : active_clusters) {
      auto vert = vertex_remapping[cluster[0]];
      cluster_verts.push_back(vert);
      cluster_vert_ids.insert(vert->id);
    }
    for (const auto &it : remapped_wall_dict) {
      const auto &vid_pair = it.first;
      const auto &wall = it.second;
      if (vid_pair.first != vid_pair.second &&
          wall->is_active &&
          cluster_vert_ids.find(vid_pair.first) != cluster_vert_ids.end() &&
          cluster_vert_ids.find(vid_pair.second) != cluster_vert_ids.end()) {
        wall_verts.push_back(std::make_pair(
            wall->left_vertex_ptr, wall->right_vertex_ptr));
      }
    }

    return std::make_shared<ProposedWorldCore>(
        wall_verts,
        cluster_verts,
        world_topology,
        vertex_remapping,
        remapped_wall_dict);
  }

  std::shared_ptr<SamplingState> get_state(double log_prob) {
    return std::make_shared<SamplingState>(
        vertices, topology, log_prob, poses);
  }

  void set_state(const std::shared_ptr<SamplingState> state) {
    topology = state->topology;

    for (const auto &vert : vertices) {
      auto it = state->vertex_info_map.find(vert->id);
      if (it != state->vertex_info_map.end()) {
        auto vert_info = it->second;
        vert->is_active = vert_info->is_active;
        vert->is_locked = vert_info->is_locked;
        vert->set_position(vert_info->position);
      }
    }

    const int num_poses(poses.size());
    for (int ii=0; ii < num_poses; ++ii) {
      auto pose = poses[ii];
      auto observation = observations[ii];
      auto it = state->pose_info_map.find(pose->index);
      if (it != state->pose_info_map.end()) {
        auto pose_info = it->second;
        pose->x = pose_info[0];
        pose->y = pose_info[1];
        pose->yaw = pose_info[2];
      }

      for (const auto &det : observation) {
        det->update_props(*pose);
      }
    }

  }

  void perform_slam_update();

  double compute_world_log_prob(
      const std::shared_ptr<ProposedWorldCore> &world,
      const std::vector<PoseObsPair> &pose_obs_pairs) const;

  double compute_world_log_prob_full(
      const std::shared_ptr<ProposedWorldCore> &world) const;
};

double compute_observation_log_likelihood(
    const std::shared_ptr<ProposedWorldCore> &world,
    const std::shared_ptr<Pose> &pose,
    const std::vector<std::shared_ptr<NoisyVertexDetection>> observation,
    const std::map<std::pair<int, int>, NoisyVertexPtr> vertex_positions_memo,
    double log_fp_rate, double log_fn_rate);
