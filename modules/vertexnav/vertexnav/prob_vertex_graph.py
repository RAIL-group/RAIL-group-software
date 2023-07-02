import itertools
import math
import numpy as np
import random
import shapely

import vertexnav
import vertexnav_accel

from .noisy import NoisyVertex

PRIOR_NOISE = np.array([0.01, 0.01, 0.01])
ODOMETRY_NOISE = np.array([.05, 0.05, 0.05])
CLUSTERING_NOISE = np.array([0.01, 0.01])


def _get_vertex_remapping(vertices, topology):
    # FIXME: does not actually work with clusters
    vertex_id_dict = {v.id: v for v in vertices}
    vertex_remapping = dict()

    for cluster in topology:
        # FIXME: not doing anything smart for now
        if False and len(cluster) == 1:
            vertex_remapping[cluster[0]] = vertex_id_dict[cluster[0]]
        else:
            # Create new noisy vertex for clusters of more than a single
            # vertex.The new NoisyVertex has a mean and covariance that
            # depends on those of its parts
            sum_cov_inv = 0
            sum_weighted_means = 0
            for cid in cluster:
                vert = vertex_id_dict[cid]
                cov_inv = np.linalg.inv(vert.cov)
                sum_cov_inv += cov_inv
                pos_T = [[vert.position[0]], [vert.position[1]]]
                sum_weighted_means += np.matmul(cov_inv, pos_T)

            cov = np.linalg.inv(sum_cov_inv)
            mean = np.matmul(cov, sum_weighted_means).tolist()

            cluster_vert = NoisyVertex(
                [mean[0][0], mean[1][0]], cov)

            for cid in cluster:
                vertex_remapping[cid] = cluster_vert

    return vertex_remapping


class Cluster(object):
    class_counter = 0

    def __init__(self, vids, vertex_id_dict):
        self.vids = vids
        self._is_active = any(vertex_id_dict[vid].is_active for vid in vids)
        self._is_locked = all(vertex_id_dict[vid].is_locked for vid in vids)
        self.vertices = [
            v for k, v in list(vertex_id_dict.items()) if k in vids
        ]
        self.last_updated = max(v.last_updated for v in self.vertices)
        self.num_dets = sum(v.num_detections for v in self.vertices)

        self.id = Cluster.class_counter
        Cluster.class_counter += 1

    @property
    def is_active(self):
        return self._is_active

    @property
    def is_locked(self):
        return self._is_locked

    def set_is_active(self, new_is_active):
        # Set state of the component vertices. Changing the state of a vertex
        # automatically unlocks it.
        for v in self.vertices:
            old_v_is_active = v.is_active
            if new_is_active is not old_v_is_active:
                v.is_active = new_is_active
                v.is_locked = False

        # Set the internal property
        self._is_active = new_is_active

    def lock(self):
        self._is_locked = True
        for v in self.vertices:
            v.is_locked = True


SamplingState = vertexnav_accel.SamplingState


class ProbVertexGraph(vertexnav_accel.ProbVertexGraph):
    def __init__(self):
        super(ProbVertexGraph, self).__init__()

        self.PRIOR_NOISE = PRIOR_NOISE
        self.ODOMETRY_NOISE = ODOMETRY_NOISE
        self.CLUSTERING_NOISE = CLUSTERING_NOISE
        self.DO_MULTI_VERTEX_MERGE = False

    def add_observation(self,
                        observation,
                        r_pose=None,
                        odom=None,
                        association_window=-1):
        """
        Add a new observation to the graph. First associate each detection
        with a single vertex. Ensure these matches are one to one. Any
        unmatched detections create new vertices. Update the graphs various
        lists and dictionaries in the process. Then pick the most likely map
        given this observation, and add the walls to the list of all walls.

        Arguments:
        self -- NoisyVertexGraph
        observation -- list of NoisyVertexDetections
        r_pose -- pose of the robot
        """
        if r_pose is not None:
            robot_id = r_pose.robot_id
        elif odom is not None:
            robot_id = odom.robot_id
        else:
            raise ValueError("One of 'r_pose' or 'odom' required.")

        num_poses_for_id = sum(1 for p in self.r_poses
                               if p.robot_id == robot_id)

        if num_poses_for_id == 0:
            if r_pose is None:
                raise ValueError("Should provide an initial pose.")
            super(ProbVertexGraph,
                  self).add_observation_pose(observation, r_pose,
                                             association_window)
        else:
            if r_pose is not None:
                raise ValueError("Cannot provide a pose after initial pose.")
            super(ProbVertexGraph,
                  self).add_observation_odom(observation, odom,
                                             association_window)

        # Run a SLAM update
        self.perform_slam_update()

    def get_proposed_world(self, r_poses=[], topology=None):
        """Get proposed world from robot poses"""
        if topology is None:
            topology = self.topology

        topology = set(tuple(c) for c in topology)

        active_verts = [
            v for v in self.vertices if v.num_detections > 2 and v.is_active
        ]
        active_verts = sorted(active_verts,
                              key=lambda v: v.last_updated,
                              reverse=True)

        # Clusters with any active vertices are active
        active_vert_ids = set((v.id for v in active_verts))
        active_clusters = set(
            cluster for cluster in topology
            if len(active_vert_ids.intersection(set(cluster))))

        # Build the temporary wall dictionary
        vertex_remapping = vertexnav_accel.get_vertex_remapping(
            self.vertices, topology)
        wall_dict = vertexnav_accel.get_remapped_wall_dict(
            self.walls, active_clusters, vertex_remapping)

        try:
            r_traj = shapely.geometry.LineString([(p.x, p.y) for p in r_poses])

            def does_intersect_robot_trajectory(wall):
                ws = shapely.geometry.LineString(
                    [wall.left_vertex.position, wall.right_vertex.position])
                return ws.intersects(r_traj)
        except ValueError as e:
            print(e)

            def does_intersect_robot_trajectory(wall):
                return False

        # Remap the clusters (for wall computation)
        remapped_clusters = [
            vertex_remapping[cluster[0]].id for cluster in active_clusters
        ]
        potential_walls = [
            wv for wk, wv in list(wall_dict.items())
            if wv.is_active and not wk[0] == wk[1]
            and wk[0] in remapped_clusters and wk[1] in remapped_clusters
            and not does_intersect_robot_trajectory(wv)
        ]

        cluster_verts = [vertex_remapping[c[0]] for c in active_clusters]
        vertex_positions = [(vertex.position[0], vertex.position[1])
                            for vertex in cluster_verts]
        wall_positions = [(wall.left_vertex.position,
                           wall.right_vertex.position)
                          for wall in potential_walls]

        return vertexnav.world.ProposedWorld(vertices=vertex_positions,
                                             walls=wall_positions,
                                             topology=topology,
                                             vertex_remapping=vertex_remapping)

    def get_known_poly(self, proposed_world, max_range=1000, topology=None):
        """Return known space polygon"""
        observations = self.observations
        r_poses = self.r_poses
        vertex_remapping = proposed_world.vertex_remapping

        h_obss = [
            vertexnav_accel.VectorHalDet(
                vertexnav.noisy.compute_hypothetical_observation(
                    proposed_world,
                    r_pose,
                    observation,
                    vertex_remapping=vertex_remapping))
            for r_pose, observation in zip(r_poses, observations)
        ]
        h_polys = [
            shapely.geometry.Polygon(
                vertexnav_accel.compute_conservative_space_hallucinated(
                    r_pose, h_obs)).buffer(
                        1e-2, resolution=0, cap_style=3, join_style=2)
            if len(h_obs) > 2 else shapely.geometry.Polygon()
            for r_pose, h_obs in zip(r_poses, h_obss)
        ]

        o_polys = [
            shapely.geometry.Polygon(
                vertexnav.noisy.compute_conservative_space_from_obs(
                    r_pose, obs, max_range,
                    vertex_remapping=vertex_remapping)).buffer(1e-2,
                                                               resolution=0,
                                                               cap_style=3,
                                                               join_style=2)
            for r_pose, obs in zip(r_poses, observations)
        ]

        poly = shapely.geometry.Polygon()
        for h_poly, o_poly in zip(h_polys, o_polys):
            poly = vertexnav.vertex_graph.safely_add_poly(
                poly, h_poly.intersection(o_poly))

        return poly.buffer(-0.9e-2, 0, cap_style=3, join_style=2)

    def sample_vertices(self,
                        p_window=15,
                        num_samples=30,
                        inflation_rad=4.0,
                        topology=None,
                        do_update_state=True,
                        active_vert_ids=None,
                        do_use_all_poses=False):
        """Consider turning on/off different vertices to sample map space"""
        if not topology:
            topology = self.topology

        active_clusters = self._get_active_clusters(
            p_window=p_window,
            inflation_rad=inflation_rad,
            topology=topology,
            active_vert_ids=active_vert_ids)
        cluster_vert_ids = set(
            itertools.chain.from_iterable(
                [list(c.vids) for c in active_clusters]))

        # Decide which pose, observation pairs are needed for prob computation.
        def does_observe_active_cluster(observation):
            for det in observation:
                if det.associated_vertex.id in cluster_vert_ids:
                    return True
            return False

        if do_use_all_poses:
            pose_obs_pairs = None
        else:
            pose_obs_pairs = [
                pair for pair in zip(self.r_poses, self.observations)
                if does_observe_active_cluster(pair[1])
            ]

        already_sampled = []

        # Store the "initial" state (needed for comparison)
        self.perform_slam_update()
        initial_proposed_world = self.get_proposed_world_fast(
            topology=self.topology)
        if do_use_all_poses:
            initial_proposed_world_log_prob = self.compute_world_log_prob_full(
                initial_proposed_world)
        else:
            initial_proposed_world_log_prob = self.compute_world_log_prob(
                initial_proposed_world, pose_obs_pairs)

        initial_sampling_state = self.get_state(
            initial_proposed_world_log_prob)

        # Set the initial updated state.
        # (if the topology is unchanged, this has no impact)
        self.topology = topology
        self.perform_slam_update()
        proposed_world = self.get_proposed_world_fast(topology=topology)
        if do_use_all_poses:
            proposed_world_log_prob = self.compute_world_log_prob_full(
                proposed_world)
        else:
            proposed_world_log_prob = self.compute_world_log_prob(
                proposed_world, pose_obs_pairs)

        best_sampling_state = self.get_state(proposed_world_log_prob)

        if len(active_clusters) == 0:
            return initial_sampling_state, best_sampling_state

        # FIXME: sort the clusters
        counter = 0
        v_iter_counter = 0
        while counter < num_samples and len(already_sampled) < len(
                active_clusters):  # noqa: E501
            # Pick a vertex and optionally toggle it
            toggle_cluster = active_clusters[v_iter_counter %
                                             len(active_clusters)]
            # toggle_cluster = random.choice(active_clusters)
            if toggle_cluster.id in already_sampled:
                v_iter_counter += 1
                continue
            elif toggle_cluster.is_locked:
                v_iter_counter += 1
                already_sampled.append(toggle_cluster.id)
                continue

            toggle_cluster.set_is_active(not toggle_cluster.is_active)

            # Compute the new proposed world and compare
            new_proposed_world = self.get_proposed_world_fast(
                topology=topology)
            if do_use_all_poses:
                new_proposed_world_log_prob = self.compute_world_log_prob_full(
                    new_proposed_world)
            else:
                new_proposed_world_log_prob = self.compute_world_log_prob(
                    new_proposed_world, pose_obs_pairs)

            if new_proposed_world_log_prob > best_sampling_state.log_prob:
                best_sampling_state = self.get_state(
                    new_proposed_world_log_prob)

            # Print the debug output
            print(
                ("#{:2d} | Init:{:8.3f} |  Best:{:8.3f} | Prop:{:8.3f} ({:2d})"
                 .format(counter, initial_proposed_world_log_prob,
                         proposed_world_log_prob, new_proposed_world_log_prob,
                         len(active_clusters))))

            # MCMC test
            delta_log_prob = new_proposed_world_log_prob \
                - proposed_world_log_prob

            if delta_log_prob > math.log(random.random() * 1.00 + 0.00):
                proposed_world = new_proposed_world
                proposed_world_log_prob = new_proposed_world_log_prob
                print("  Switching")
                already_sampled = []
                v_iter_counter = 0
            else:
                # Switch the cluster back
                toggle_cluster.set_is_active(not toggle_cluster.is_active)
                # if delta_log_prob < -100:
                #     print("  Locking cluster: ")
                #     toggle_cluster.lock()

                already_sampled.append(toggle_cluster.id)

            v_iter_counter += 1
            counter += 1

        # Now either restore the PVG to its original state or use the maximum
        # likelihood state.
        if do_update_state:
            self.set_state(best_sampling_state)
        else:
            self.set_state(initial_sampling_state)

        return initial_sampling_state, best_sampling_state

    def sample_states(self,
                      num_topology_samples=3,
                      num_vertex_samples=30,
                      vertex_association_dist_threshold=10,
                      vertex_association_time_window=15,
                      vertex_sampling_time_window=15,
                      vertex_visibility_inflation_rad=4.0,
                      mvmp_merge_dist_threshold=0.5,
                      do_update_state=True):
        """Primary sampling functionality. Samples over topologies (if
        DO_SAMPLE_TOPOLOGY is True) and over vertices."""
        if self.DO_SAMPLE_TOPOLOGY:
            if self.DO_MULTI_VERTEX_MERGE:
                print("Sampling topologies (with multi vertex merge)")
            else:
                print("Sampling topologies (with single vertex merge)")

            return self.sample_topologies(
                num_samples=num_topology_samples,
                association_dist_threshold=vertex_association_dist_threshold,
                vertex_association_time_window=vertex_association_time_window,
                vertex_sampling_time_window=vertex_sampling_time_window,
                num_vertex_samples=num_vertex_samples,
                inflation_rad=vertex_visibility_inflation_rad,
                do_update_state=do_update_state,
                mvmp_merge_dist_threshold=mvmp_merge_dist_threshold,
                do_multi_vertex_merge=self.DO_MULTI_VERTEX_MERGE)
        else:
            print("Sampling vertices only (with no vertex merge)")
            return self.sample_vertices(
                p_window=vertex_sampling_time_window,
                num_samples=num_vertex_samples,
                inflation_rad=vertex_visibility_inflation_rad,
                do_update_state=do_update_state)

    def sample_topologies(self,
                          num_samples=3,
                          association_dist_threshold=10,
                          vertex_association_time_window=15,
                          vertex_sampling_time_window=15,
                          num_vertex_samples=30,
                          inflation_rad=4.0,
                          mvmp_merge_dist_threshold=0.5,
                          do_update_state=True,
                          do_multi_vertex_merge=False):

        # First sample vertices with the current topology
        initial_state, best_state = self.sample_vertices(
            p_window=vertex_sampling_time_window,
            num_samples=num_vertex_samples,
            inflation_rad=inflation_rad,
            do_update_state=True)

        # Helper variables
        vertex_id_dict = {v.id: v for v in self.vertices}
        active_clusters = self._get_active_clusters(
            p_window=vertex_association_time_window,
            inflation_rad=inflation_rad,
            topology=self.topology)
        active_vert_ids = set(
            itertools.chain.from_iterable(
                [list(c.vids) for c in active_clusters]))

        successful_samples = 0
        did_sample = []

        for ii in range(2000):
            # Generate  new topology from the current topology
            new_topology, _, move_type = vertexnav.utils.topology.draw_sample(
                self.topology)

            # Compute which vertices changed via the topology update
            old_t = frozenset([frozenset(c) for c in self.topology])
            new_t = frozenset([frozenset(c) for c in new_topology])
            updated_vert_ids = list(
                itertools.chain.from_iterable(old_t - new_t))
            if len(updated_vert_ids) == 0:
                continue

            # Decide if the new topology is valid
            # If any of the vertices in the updated_vert_ids are stale, reject.
            if len(set(updated_vert_ids) - active_vert_ids):
                continue

            # At least one vertex should be within p_window
            # if vertex_association_time_window is not None \
            #    and vertex_association_time_window < len(self.r_poses):
            #     last_updated_thresh = self.r_poses[
            #         len(self.r_poses) - vertex_association_time_window].index
            #     if not any(
            #             vertex_id_dict[vid].last_updated >= last_updated_thresh
            #             for vid in updated_vert_ids):
            #         continue

            # # If any of the vertices have basically no detections
            # if any(vertex_id_dict[vid].num_detections <= 10
            #        for vid in updated_vert_ids):
            #     continue

            # During a merge, if any of the vertices
            if move_type == 1:  # merge

                def eucl_dist_sq(vida, vidb):
                    va = vertex_id_dict[vida]
                    vb = vertex_id_dict[vidb]
                    return ((va.position[0] - vb.position[0])**2 +
                            (va.position[1] - vb.position[1])**2)

                association_dist_thresh_sq = association_dist_threshold**2

                if any(
                        eucl_dist_sq(vid_a, vid_b) > association_dist_thresh_sq
                        for vid_a, vid_b in itertools.permutations(
                            updated_vert_ids, 2)):
                    continue

            # Check if we've tried this one already or if it's disallowed
            top_diff = (set(tuple(sorted(c)) for c in (old_t - new_t)),
                        set(tuple(sorted(c)) for c in (new_t - old_t)))
            # diff = (old_t, new_t)
            if top_diff in self.disallowed_topology_operations:
                continue
            elif top_diff in did_sample:
                continue
            else:
                did_sample += [top_diff]

            # Count the number of times a sample was allowed to be tested.
            successful_samples += 1

            # Sample with the new topology
            initial_trial_state, trial_state = self.sample_vertices(
                topology=new_topology,
                do_update_state=False,
                p_window=vertex_sampling_time_window,
                num_samples=num_vertex_samples,
                inflation_rad=inflation_rad,
                active_vert_ids=updated_vert_ids,
                do_use_all_poses=True)

            # If we already did a single vertex merge, and MVMP is in use, then
            # we attempt to merge again
            if move_type == 1 and do_multi_vertex_merge:
                # Compute new topolology corresponding to a multi-vertex merge
                mvmp_topology = []

                # *Temporarily* set the PVG state to trial state
                # NOTE: must be _reset_ after evaluation
                self.set_state(trial_state)

                # TODO(kevin): Manually do topological surgery
                vertex_id_dict = {v.id: v for v in self.vertices}

                mvmp_topology = []
                merged_set = set()

                # Propose new merges
                for ci in self.topology:
                    vidi = ci[0]
                    cij = list(ci)
                    if tuple(ci) in merged_set:
                        continue
                    merged_set.add(tuple(ci))
                    for cj in self.topology:
                        if tuple(cj) in merged_set:
                            continue
                        vidj = cj[0]

                        def eucl_dist_sq(vida, vidb):
                            va = vertex_id_dict[vida]
                            vb = vertex_id_dict[vidb]
                            return ((va.position[0] - vb.position[0])**2 +
                                    (va.position[1] - vb.position[1])**2)

                        if eucl_dist_sq(vidi, vidj) < mvmp_merge_dist_threshold**2:
                            cij += list(cj)
                            merged_set.add(tuple(cj))

                    mvmp_topology.append(cij)

                assert(len(sum(mvmp_topology, [])) == len(sum(self.topology, [])))

                # Reconstruct clusters
                # cluster_map += [c for c in self.topology if c not in merged_set]

                # Ensure each vertex id _does not_ appear more than once
                # TODO(kevin): assert ???

                # for v in self.vertices:
                #     if v.position[0] > -0.2 and v.position[1] < 0.2:
                #         origin_verts += [v]
                #     else:
                #         topology += [(v.id, )]

                # Compute which vertices changed via the topology update
                old_t = frozenset([frozenset(c) for c in self.topology])
                new_t = frozenset([frozenset(c) for c in mvmp_topology])
                updated_vert_ids = list(
                    itertools.chain.from_iterable(old_t - new_t))

                # Sample with the MVMP topology
                _, mvmp_trial_state = self.sample_vertices(
                    topology=mvmp_topology,
                    do_update_state=False,
                    p_window=vertex_sampling_time_window,
                    num_samples=num_vertex_samples,
                    inflation_rad=inflation_rad,
                    active_vert_ids=updated_vert_ids,
                    do_use_all_poses=True)

                if mvmp_trial_state.log_prob > trial_state.log_prob:
                    trial_state = mvmp_trial_state

                # Reset PVG state to current best after evaluating MVMP
                self.set_state(best_state)

            # MCMC criteria
            d_prob_trial = trial_state.log_prob - initial_trial_state.log_prob
            # Is this correct?
            # d_prob_trial = trial_state.log_prob - best_state.log_prob
            if d_prob_trial > math.log(random.random() * 1.00 + 0.00):
                print(
                    "Switching: \n" +
                    "  New State Prob: {}\n".format(trial_state.log_prob) +
                    "  Initial State Prob: {}\n".format(
                        initial_trial_state.log_prob) +
                    "  Old Best State Prob: {}\n".format(best_state.log_prob))
                did_sample = []
                self.set_state(trial_state)
                best_state = trial_state
            else:
                print("Not Switching: \n" +
                      "  (rejected) Trial State Prob: {}\n".format(
                          trial_state.log_prob) +
                      "  Initial State Prob: {}\n".format(
                          initial_trial_state.log_prob) +
                      "  Best State Prob: {}\n".format(best_state.log_prob))

            if successful_samples > num_samples:
                break

        return initial_state, best_state

    def _get_active_clusters(self,
                             p_window,
                             inflation_rad,
                             topology,
                             active_vert_ids=None):
        """Helper function to compute which clusters may be flipped
        during sampling."""

        if active_vert_ids is None:
            observations = self.observations[-p_window:]
            r_poses = self.r_poses[-p_window:]
            polys = [
                shapely.geometry.Polygon(
                    vertexnav.noisy.compute_conservative_space_from_obs(
                        r_pose, obs)).buffer(inflation_rad,
                                             resolution=0,
                                             cap_style=3,
                                             join_style=2)
                for r_pose, obs in zip(r_poses, observations)
            ]

            def is_inside_polys(position):
                for poly in polys:
                    if poly.contains(shapely.geometry.Point(position)):
                        return True
                return False

            # Compute which vertices may be toggled
            seen_verts = {
                v
                for v in self.vertices if is_inside_polys(v.position)
            }
            recent_verts = {
                det.associated_vertex
                for obs in observations for det in obs
            }
            verts = [
                v for v in seen_verts.union(recent_verts)
                if v.num_detections > 2 and not v.is_locked
            ]
            vert_ids = [v.id for v in verts]
            vert_ids = set(vert_ids)
        else:
            vert_ids = set(active_vert_ids)

        # Compute which clusters may be toggled
        vertex_id_dict = {v.id: v for v in self.vertices}
        clusters = [Cluster(c, vertex_id_dict) for c in topology]
        active_clusters = [
            c for c in clusters if not vert_ids.isdisjoint(c.vids)
        ]
        active_clusters = sorted(active_clusters,
                                 key=lambda c: c.last_updated,
                                 reverse=True)

        # No elements should be in the vert_ids that are not part of a cluster.
        cluster_vert_ids = set(
            itertools.chain.from_iterable(
                [list(c.vids) for c in active_clusters]))
        assert len(vert_ids.difference(cluster_vert_ids)) == 0

        active_clusters.sort(reverse=True, key=lambda c: c.last_updated)

        return active_clusters
