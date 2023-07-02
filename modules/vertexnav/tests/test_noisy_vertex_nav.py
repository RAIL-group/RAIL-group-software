import vertexnav
from vertexnav.world import World
import numpy as np
import pytest
import random

from shapely import geometry


def get_world_square(is_inside=False):
    # A square
    square_poly = geometry.Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    if is_inside:
        return World(obstacles=[], boundary=square_poly)
    else:
        return World(obstacles=[square_poly])


def get_world_two_squares_vert():
    # A square
    square_poly_1 = geometry.Polygon([(-1, 1), (-1, 2), (1, 2), (1, 1)])
    square_poly_2 = geometry.Polygon([(-1, -2), (-1, -1), (1, -1), (1, -2)])

    return World(obstacles=[square_poly_1, square_poly_2])


def get_world_two_squares_horz():
    # A square
    square_poly_1 = geometry.Polygon([(-1, -1), (-1, 1), (-2, 1), (-2, -1)])
    square_poly_2 = geometry.Polygon([(1, -1), (1, 1), (2, 1), (2, -1)])

    return World(obstacles=[square_poly_1, square_poly_2])


def test_prob_of_wall():
    """Tests that the most likely map corresponds to the one with the walls in the correct places"""
    pytest.xfail("Test not yet written.")

    world_sq = get_world_square()
    world_vert = get_world_two_squares_vert()
    world_horz = get_world_two_squares_horz()
    worlds = []
    worlds.append(world_sq)
    worlds.append(world_vert)
    worlds.append(world_horz)


def test_square_map_building_no_noise():
    """Tests that we can build a simple square map by fusing multiple measurements
    (without any false positive or negative detections)."""
    world = get_world_square()
    nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()

    counter = 0
    for ii in range(100):
        robot_pose = vertexnav.Pose(x=0.001, y=0.001, yaw=0.0)
        counter += 1
        obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            world.get_vertices_for_pose(robot_pose),
            robot_pose,
            do_add_noise=False,
            cov_rt=[[0.5, 0], [0, 0.5]])
        if ii == 0:
            nvg.add_observation(obs, robot_pose)
        else:
            nvg.add_observation(obs, odom=vertexnav.Pose(0, 0, 0))
        assert (len(obs) == 4)

    assert (len(nvg.vertices) == 4)

    # Confirm that the number of "active" walls is also 4
    wall_count = len([
        None for k, wall in nvg.walls.items()
        if wall.is_active and not k[0] == k[1]
    ])

    assert (wall_count == 4)

    nvg.sample_vertices(num_samples=20, p_window=1000, inflation_rad=0.1)
    proposed_world = nvg.get_proposed_world_fast(nvg.topology)
    assert (len(nvg.vertices) == 4)
    assert (len(proposed_world.vertices) == 4)
    assert (nvg.compute_world_log_prob(
        proposed_world, list(zip(nvg.r_poses, nvg.observations))) < 0)


def test_square_map_building_position_noise():
    """Tests that we can build a simple square map by fusing multiple measurements
    (without any false positive or negative detections)."""
    world = get_world_square(is_inside=True)
    nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()

    for ii in range(10):
        robot_pose = vertexnav.Pose(x=0.0, y=0.0, yaw=0.0)
        obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            world.get_vertices_for_pose(robot_pose),
            robot_pose,
            do_add_noise=True,
            cov_rt=[[0.5, 0], [0, 0.5]])
        if ii == 0:
            nvg.add_observation(obs, robot_pose)
        else:
            nvg.add_observation(obs, odom=vertexnav.Pose(0, 0, 0))

    assert (len(nvg.vertices) == 4)

    # Confirm that the number of "active" walls is also 4
    wall_count = len([
        None for k, wall in nvg.walls.items()
        if wall.is_active and not k[0] == k[1]
    ])

    assert (wall_count == 4)


@pytest.mark.parametrize("fp_range", [(2.0), (0.5)])
def test_square_map_building_sampling_false_positive(fp_range):
    """Show that, even with a couple false-positive detections,
    we can get the map that best corresponds to the real map."""

    # Initializations
    world = get_world_square(is_inside=True)
    nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()

    # Construct the noisy graph with a handful of measurements
    prev_pose = None
    counter = 0
    for _ in range(25):
        robot_pose = vertexnav.Pose(x=0.0, y=0.0, yaw=0.0)
        counter += 1
        obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            world.get_vertices_for_pose(robot_pose),
            robot_pose,
            do_add_noise=True,
            cov_rt=[[1, 0], [0, 0.5]])
        if prev_pose is not None:
            odom = vertexnav.Pose.get_odom(
                p_new=vertexnav.Pose(robot_pose.x, robot_pose.y,
                                     robot_pose.yaw),
                p_old=vertexnav.Pose(prev_pose.x, prev_pose.y, prev_pose.yaw))
            nvg.add_observation(obs, odom=odom)
        else:
            nvg.add_observation(obs, robot_pose)
        prev_pose = robot_pose

    # Now add some systematically 'bad' measurements
    for _ in range(5):
        robot_pose = vertexnav.Pose(x=0.0, y=0.0, yaw=0.0)
        counter += 1
        obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            world.get_vertices_for_pose(robot_pose),
            robot_pose,
            do_add_noise=True,
            cov_rt=[[1, 0], [0, 0.5]])
        obs.append(
            vertexnav.noisy.NoisyVertexDetection(
                angle_rad=0.0,
                range=fp_range,
                detection_type=vertexnav.noisy.NoisyDetectionType(
                    [0.25, 0.25, 0.25, 0.25]),
                cov_rt=np.array([[1, 0], [0, 0.5]])))
        obs = sorted(obs, key=lambda pvd: pvd.angle_rad)
        odom = vertexnav.Pose.get_odom(
            p_new=vertexnav.Pose(robot_pose.x, robot_pose.y, robot_pose.yaw),
            p_old=vertexnav.Pose(prev_pose.x, prev_pose.y, prev_pose.yaw))
        nvg.add_observation(obs, odom=odom)
        prev_pose = robot_pose

    # Now get the 'most likely' proposed world
    random.seed(1234)
    np.random.seed(1234)
    old_proposed_world = nvg.get_proposed_world_fast(topology=nvg.topology)
    old_prob = nvg.compute_world_log_prob(
        old_proposed_world, list(zip(nvg.r_poses, nvg.observations)))
    nvg.sample_vertices(num_samples=20, p_window=1000, inflation_rad=0.1)
    proposed_world = nvg.get_proposed_world()
    assert (len(nvg.vertices) == 5)
    assert (len(proposed_world.vertices) == 4)
    assert (len(proposed_world.walls) == 4)
    proposed_world = nvg.get_proposed_world_fast(topology=nvg.topology)
    new_prob = nvg.compute_world_log_prob(
        proposed_world, list(zip(nvg.r_poses, nvg.observations)))

    assert new_prob < 0
    assert new_prob >= old_prob


def test_get_m_distance():
    pos_a = [0, 0]
    pos_b = [2, 0]
    pos_c = [4, 0]
    pos_d = [0, 3]

    cov_1 = np.array([[4, 0], [0, 9]])
    inv_cov_1 = np.linalg.inv(cov_1)

    assert vertexnav.utils.calc.m_distance(pos_a, pos_b,
                                           inv_noise=inv_cov_1) == 1.0
    assert vertexnav.utils.calc.m_distance(pos_a, pos_c,
                                           inv_noise=inv_cov_1) == 2.0
    assert vertexnav.utils.calc.m_distance(pos_a, pos_d,
                                           inv_noise=inv_cov_1) == 1.0


@pytest.mark.parametrize("r_cov, theta, dist, r_theta",
                         [(4.0, 0.0, 1.0, 0.01), (4.0, 0.0, 2.0, 0.01),
                          (4.0, 1.0, 1.0, 0.01), (4.0, 1.0, 2.0, 0.01),
                          (2.0, 2.0, 2.0, 0.01)])
def test_transform_local_to_world(r_cov, theta, dist, r_theta):
    _, cov = vertexnav.utils.calc.transform_local_to_world(
        det_theta=theta,
        det_r=dist,
        cov_rt=np.array([[r_cov, 0], [0, r_theta]]),
        r_pose=vertexnav.Pose(0, 0))

    # Confirm that the max eigenvalue of the transformed covariance matrix
    # matches that of the original matrix.
    vals, vecs = np.linalg.eigh(cov)
    assert max(vals) == pytest.approx(r_cov)
    assert min(vals) == pytest.approx(r_theta * dist * dist)


def test_detect_peaks():
    """Confirm the peaks are where we expect."""
    grid = np.zeros([100, 100]) * 1.0
    known_coords = np.array([
        [9, 14],
        [9, 48],
        [25, 14],
        [99, 20],
    ])

    for c in known_coords:
        grid[c[0], c[1]] = 1.0

    peaks = vertexnav.utils.calc.detect_peaks(grid, peak_thresh=0.5)

    assert len(peaks) == len(known_coords)
    for p in peaks:
        assert p in known_coords
