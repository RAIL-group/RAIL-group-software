import math
import numpy as np
import scipy
import shapely

import vertexnav
from .noisy import NoisyVertex, NoisyWall


def safely_add_poly(poly, adding_poly):
    """Combine two shapely (or other type) polygons"""
    try:
        new_poly = poly.union(adding_poly)
        if isinstance(new_poly, shapely.geometry.GeometryCollection):

            new_new_poly = poly
            for geom in new_poly:
                if isinstance(geom, shapely.geometry.Polygon):
                    new_new_poly = new_new_poly.union(geom)
                else:
                    new_new_poly = new_new_poly.union(
                        geom.buffer(1e-2,
                                    resolution=0,
                                    cap_style=3,
                                    join_style=2))

                if isinstance(new_new_poly, shapely.geometry.Polygon):
                    new_poly = new_new_poly
                else:
                    return poly
        if new_poly.is_valid and new_poly.area >= 0.9999 * poly.area:
            return new_poly.simplify(1.5e-2, preserve_topology=False)
        else:
            return poly

    except Exception as e:
        print("Failure")
        print((str(e)))
        return poly


class PerfectVertexGraph(object):
    """
    PerfectVertexGraph represents the underlying map of vertices and wall edges.
    Same as Noisy version, but with (obviously) no noise in observations
    """
    def __init__(self):
        self.vertices = []
        self.walls = {}  # Dict: {(vert_L.id, vert_R.id) : NoisyWall}
        self.r_poses = []
        self.observations = []
        self.world = None

        self.false_positive_rate = 0.0
        self.false_negative_rate = 0.0

        self._last_proposed_world = None
        self._known_space_last_updated = -1
        self._known_space_poly = shapely.geometry.Polygon([])

    def add_wall_detection(self,
                           vert_det_L,
                           vert_det_R,
                           num_false_positives=0):
        """Updates walls to include sensor observations. This function
        requires that the 'inverse_associations' has already been taken
        care of."""

        # Check that the inverse association has not yet been done
        assert not isinstance(vert_det_L, NoisyVertex)
        assert not isinstance(vert_det_R, NoisyVertex)

        # If the angle spread is too high, reject this wall proposal
        dtheta = (vert_det_L.angle_rad - vert_det_R.angle_rad) % (2 * math.pi)
        if dtheta >= math.pi:
            return

        vert_L = vert_det_L.associated_vertex
        vert_R = vert_det_R.associated_vertex

        key = tuple(sorted([vert_L.id, vert_R.id]))
        if key not in self.walls:
            self.walls[key] = NoisyWall(vert_L, vert_R)

        fp_factor = self.false_positive_rate**num_false_positives
        self.walls[key].add_detection(vert_det_L,
                                      vert_det_R,
                                      false_positive_factor=fp_factor)

    def add_observation(self, observation, r_pose):
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
        for det in observation:
            det.update_props(r_pose)

        self.r_poses.append(r_pose)
        self.observations.append(observation)
        self.associate_detections(observation, r_pose)

        # Add the wall detections
        shift = 0
        shifted_observation = observation
        while shift < len(observation):
            shifted_observation = shifted_observation[
                1:] + shifted_observation[:1]
            shift += 1
            for det_L, det_R in zip(shifted_observation, observation):
                self.add_wall_detection(det_L,
                                        det_R,
                                        num_false_positives=shift - 1)

    def associate_detections(self, observation, r_pose):
        """Use an optimal assignment to compute the vertex association.
        Priority is given to the active vertices"""

        # Updating the vertices invalidates the vertex positions memo
        self.vertex_positions_memo = None

        # First assignment
        all_verts = self.vertices
        all_detections = observation
        if len(all_detections) == 0:
            return

        # Orer: detections, vertices (+ unassociations)
        cost_mat = 1.0 * np.ones(
            (len(all_detections), len(all_verts) + len(all_detections)))
        for dind, det in enumerate(all_detections):
            det_inv_cov = np.linalg.inv(det.cov)
            for vind, vert in enumerate(all_verts):
                cost_mat[dind, vind] = vertexnav.utils.calc.m_distance(
                    vert.position, det.position, inv_noise=det_inv_cov)
                if not vert.is_active:
                    cost_mat[dind, vind] += 0.8

        dinds, vinds = scipy.optimize.linear_sum_assignment(cost_mat)

        unassociated_detections = []
        for dind, vind in zip(dinds, vinds):
            # Check to see if it's unassociated
            if vind >= len(all_verts):
                unassociated_detections.append(all_detections[dind])
            else:
                vert = all_verts[vind]
                det = all_detections[dind]
                vert.add_detection(det.position, det.cov, r_pose)
                det.add_association(vert)
        if len(unassociated_detections) == 0:
            return

        unassociated_verts = [
            v for ii, v in enumerate(all_verts) if ii not in vinds
        ]

        # Orer: detections, vertices (+ unassociations)
        cost_mat = 1.0 * np.ones(
            (len(unassociated_detections),
             len(unassociated_verts) + len(unassociated_detections)))
        for dind, det in enumerate(unassociated_detections):
            det_inv_cov = np.linalg.inv(det.cov)
            for vind, vert in enumerate(unassociated_verts):
                cost_mat[dind, vind] = vertexnav.utils.calc.m_distance(
                    vert.position, det.position, inv_noise=det_inv_cov)

        dinds, vinds = scipy.optimize.linear_sum_assignment(cost_mat)

        for dind, vind in zip(dinds, vinds):
            # Check to see if it's unassociated
            det = unassociated_detections[dind]
            if vind >= len(unassociated_verts):
                # Create a new vertex
                new_vert = NoisyVertex(det.position, det.cov)
                det.add_association(new_vert)
                self.vertices.append(new_vert)
            else:
                # Add association
                vert = unassociated_verts[vind]
                vert.add_detection(det.position, det.cov, r_pose)
                det.add_association(vert)

    def get_proposed_world(self, r_poses=[], num_detections_lower_limit=1):
        print("Limiting active verts")
        active_verts = [
            v for v in self.vertices
            if v.num_detections > num_detections_lower_limit
        ]

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

        potential_walls = [
            w for w in list(self.walls.values())
            if w.prob_exists > 0.01 and not does_intersect_robot_trajectory(w)
        ]
        potential_walls = sorted(
            potential_walls,
            key=lambda w: max(w.left_vertex.last_updated, w.right_vertex.
                              last_updated),
            reverse=True)

        vertex_positions = [(vertex.position[0], vertex.position[1])
                            for vertex in active_verts]
        wall_positions = [(wall.left_vertex.position,
                           wall.right_vertex.position)
                          for wall in potential_walls]

        proposed_world = vertexnav.world.ProposedWorld(
            vertices=vertex_positions, walls=wall_positions)

        old_proposed_world = self._last_proposed_world
        if old_proposed_world is not None:
            if len(vertex_positions) == len(old_proposed_world.vertices) \
               and len(wall_positions) == len(old_proposed_world.walls):
                return self._last_proposed_world
            else:
                # Old obstacles still exist, so keep the memos from
                # covisibility queries that returned False
                proposed_world.covis_memo = {
                    k: v
                    for k, v in old_proposed_world.covis_memo.items() if not v
                }

        self._last_proposed_world = proposed_world
        return proposed_world

    def get_known_poly(self, proposed_world, max_range=1000):
        pose_obs = [(r_pose, obs)
                    for r_pose, obs in zip(self.r_poses, self.observations)
                    if r_pose.index > self._known_space_last_updated]

        polys = [
            shapely.geometry.Polygon(
                vertexnav.noisy.compute_conservative_space_from_obs(
                    r_pose, obs, max_range)).buffer(1e-2,
                                                    resolution=0,
                                                    cap_style=3,
                                                    join_style=2)
            for r_pose, obs in pose_obs
        ]

        for poly in polys:
            self._known_space_poly = safely_add_poly(self._known_space_poly,
                                                     poly)

        self._known_space_last_updated = self.r_poses[-1].index
        return self._known_space_poly.buffer(-1e-2,
                                             1,
                                             cap_style=3,
                                             join_style=2)
