import math
import numpy as np

import vertexnav
import vertexnav_accel
import shapely
"""
- Some terminology:
- Vertex is a landmark; a point that defines part of a polygonal boundary
- Detection refers to a single measurement of a landmark
- Observation refers to a set of detections from a single viewpoint; an
    observation should be ordered by the image-space 'angle' of the detections.
"""

NoisyDetectionType = vertexnav_accel.NoisyDetectionType
NoisyVertexDetection = vertexnav_accel.NoisyVertexDetection
HallucinatedVertexDetection = vertexnav_accel.HallucinatedVertexDetection
NoisyVertex = vertexnav_accel.NoisyVertex
NoisyWall = vertexnav_accel.NoisyWall


def compute_conservative_space_from_obs(r_pose,
                                        observation,
                                        radius=1000,
                                        vertex_remapping=None):
    """Returns a list of points corresponding to the known space polygon
(conservative estimate) from an observation."""

    if not observation or len(observation) == 1:
        return []

    if isinstance(observation[0], HallucinatedVertexDetection):
        # Remapping the vertices is not necessary (and will fail) for
        # HallucinatedVertexDetection objects.
        def get_position(det):
            return det.position
    elif isinstance(observation[0], NoisyVertexDetection):
        if vertex_remapping is None:

            def get_position(det):
                return det.associated_vertex.position
        else:

            def get_position(det):
                return vertex_remapping[det.associated_vertex.id].position
    else:
        raise NotImplementedError()

    positions = [get_position(det) for det in observation]
    dat = [(p, math.atan2(p[1] - r_pose.y, p[0] - r_pose.x))
           for p in positions]

    dat = sorted(dat, key=lambda d: d[1])

    shift_dat = dat[1:] + dat[:1]

    poly_points = []
    for (_, high_angle), (pos_R, low_angle) in zip(shift_dat, dat):
        if high_angle < low_angle:
            high_angle += 2 * math.pi

        poly_points.append(pos_R)
        if high_angle - low_angle >= math.pi:
            poly_points.append((r_pose.x, r_pose.y))

    poly_points.append(poly_points[0])
    return poly_points


class HypotheticalObservation(list):
    def __hash__(self):
        raise NotImplementedError()


def compute_hypothetical_observation(world,
                                     r_pose,
                                     observation,
                                     radius=1000,
                                     vertex_remapping=None):
    """Compute the hypothesized observation given a proposed world, a robot pose, and the actual observation at that point.

The hypothesized observation should not introduce any new visibility
connections, so the actual observation is used to prune points that were not
observed in the space."""
    # Compute the polygon formed by the observation
    poly_points = vertexnav_accel.compute_conservative_space_observed(
        r_pose, observation, vertex_remapping)

    if isinstance(world, vertexnav_accel.ProposedWorldCore):
        h_vertex_data = world.get_vertices_for_pose(r_pose, poly_points)
    else:
        h_vertex_data = world.get_vertices_for_pose(
            r_pose, do_compute_detection_type=False, bound_points=poly_points)
    if not h_vertex_data:
        return HypotheticalObservation([])

    positions = np.array(h_vertex_data)
    ds = positions - [r_pose.x, r_pose.y]
    angles = np.arctan2(ds[:, 1], ds[:, 0])

    h_obs = [
        HallucinatedVertexDetection(angle_rad=angle_rad, position=position)
        for (position, angle_rad) in zip(
            positions,
            angles,
        )
    ]
    h_obs = sorted(h_obs, key=lambda d: d.angle_rad)

    return HypotheticalObservation(h_obs)


def get_angle_rad(position, obs_pose):
    return math.atan2(position[1] - obs_pose.y, position[0] - obs_pose.x)


def get_range(position, obs_pose):
    return math.sqrt((position[1] - obs_pose.y)**2 +
                     (position[0] - obs_pose.x)**2)


def prob_of_wall(label1, label2):
    # return (1 - label1.prob_left_gap) * (1 - label2.prob_right_gap)
    return (1 - 0.5 * (label1.prob_left_gap + label2.prob_right_gap))


def convert_world_obs_to_noisy_detection(world_obs,
                                         pose,
                                         do_add_noise=False,
                                         cov_rt=None):
    """Helper function. Convert the output of 'world.get_vertices_for_pose into
    the form necessary for working with NoisyVertexGraph (NoisyVertexDetection).
    FIXME(gjstein): need to add covariance.
    """
    def get_angle_rad(position, obs_pose):
        return math.atan2(position[1] - obs_pose.y,
                          position[0] - obs_pose.x) - obs_pose.yaw

    if do_add_noise:
        if cov_rt is None:
            raise ValueError("cov_rt must be set when do_add_noise is True")

        noise = np.random.normal(scale=0.01, size=[2])
    else:
        noise = np.zeros([2])

    if cov_rt is None:
        obs = [
            HallucinatedVertexDetection(
                angle_rad=get_angle_rad(position + noise, pose),
                range=get_range(position + noise, pose),
                r_pose=pose) for position, detection_label in world_obs
        ]
        for det, (_, detection_label) in zip(obs, world_obs):
            det.detection_type = vertexnav.noisy.NoisyDetectionType(
                detection_label)
    else:
        obs = [
            NoisyVertexDetection(
                angle_rad=get_angle_rad(position + noise, pose),
                range=get_range(position + noise, pose),
                detection_type=NoisyDetectionType(detection_label),
                cov_rt=cov_rt) for position, detection_label in world_obs
        ]

    return sorted(obs, key=lambda pvd: pvd.angle_rad)


def get_visibility_polygon(pose, world, radius=100, is_conservative=True):
    """Helper function. Returns the points on the boundary of the visibility
    polygon for a given pose. Conservative flag should be kept true.
    """
    vertices = world.get_vertices_for_pose(pose)
    observation = convert_world_obs_to_noisy_detection(vertices, pose)

    if not vertices:
        return [(radius * math.cos(th) + pose.x,
                 radius * math.sin(th) + pose.y)
                for th in np.linspace(0, 2 * math.pi, num=6)]

    shifted_observation = observation[1:] + observation[:1]

    new_walls = []
    for det in observation:
        th = det.angle_rad
        new_walls.append((det.position, (radius * math.cos(th) + pose.x,
                                         radius * math.sin(th) + pose.y)))

    poly_points = []
    for det_L, det_R in zip(shifted_observation, observation):

        # Get angles for sweep
        low_angle, high_angle = det_R.angle_rad, det_L.angle_rad
        if high_angle < low_angle:
            high_angle += 2 * math.pi

        is_wall = prob_of_wall(det_R.detection_type,
                               det_L.detection_type) > 0.5

        does_go_around = False
        if high_angle - low_angle >= math.pi:
            # observation goes around robot
            does_go_around = True

        if is_wall or (is_conservative and not does_go_around):
            poly_points.append(det_R.position)
            poly_points.append(det_L.position)
        else:
            for th in np.linspace(low_angle, high_angle, num=3):
                poly_points.append((radius * math.cos(th) + pose.x,
                                    radius * math.sin(th) + pose.y))

    poly_points.append(poly_points[0])

    if is_conservative:
        return poly_points

    point = shapely.geometry.Point(pose.x, pose.y)
    polygon = shapely.geometry.Polygon(poly_points)

    def get_angle_rad(position, obs_pose):
        return math.atan2(position[1] - obs_pose.y, position[0] - obs_pose.x)

    for wall in world.walls:
        angles = (get_angle_rad(wall[0], pose), get_angle_rad(wall[1], pose))

        ws = shapely.geometry.LineString(
            ((2 * radius * math.cos(angles[0]) + pose.x,
              2 * radius * math.sin(angles[0]) + pose.y), wall[0], wall[1],
             (2 * radius * math.cos(angles[1]) + pose.x,
              2 * radius * math.sin(angles[1]) + pose.y)))
        try:
            result = shapely.ops.split(polygon, ws)
        except:  # noqa
            result = [polygon]
        for poly in result:
            if poly.contains(point):
                polygon = poly

    poly_points = list(zip(*polygon.boundary.xy))

    return poly_points


def convert_net_grid_data_to_noisy_detection(net_data, pose, max_range,
                                             num_range, num_bearing, sig_r,
                                             sig_th, nn_peak_thresh):
    """Helper function. Convert the output of 'world.get_vertices_for_pose into
    the form necessary for working with NoisyVertexGraph (NoisyVertexDetection).
    """
    # Lookup vectors
    vec_range, vec_bearing = vertexnav.utils.calc.range_bearing_vecs(
        max_range, num_range, num_bearing)
    out_coords = vertexnav.utils.calc.detect_peaks(net_data['is_vertex'],
                                                   is_circular_boundary=True,
                                                   peak_thresh=nn_peak_thresh)

    obs = [
        NoisyVertexDetection(angle_rad=vec_bearing[ith],
                             range=vec_range[ir],
                             detection_type=NoisyDetectionType(
                                 list(net_data['vertex_label'][ir, ith])),
                             cov_rt=[[sig_r**2, 0], [0, sig_th**2]])
        for (ir, ith) in out_coords
    ]

    return sorted(obs, key=lambda pvd: pvd.angle_rad)
