import vertexnav
import math
import numpy as np
import pytest
from shapely import geometry


def test_proposed_world_dist():
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0), (1, 0.4),
                                                    (-1, 0.4), (0, 1)],
                                          walls=[[(0, 0), (1, 0.4)],
                                                 [(0, 0), (-1, 0.4)],
                                                 [(0, 0), (0, 1)]])

    Pose = vertexnav.Pose
    assert world.get_dist(Pose(0, 0)) == pytest.approx(0)
    assert world.get_dist(Pose(0, -1)) == pytest.approx(1)


def test_vis_graph_path_gen():
    world = vertexnav.world.ProposedWorld(vertices=[(0, -1), (0, 1)],
                                          walls=[[(0, -1), (0, 1)]])
    vgraph = vertexnav.planning.VisibilityGraph(world)

    path = vgraph.get_shortest_path(start_point=(-1, 0), end_point=(1, 0))

    def path_length(path):
        return sum([
            math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            for p1, p2 in zip(path[:-1], path[1:])
        ])

    assert path[0] == (-1, 0)
    assert path[-1] == (1, 0)
    assert path_length(path) >= 2 * math.sqrt(2)
    assert path_length(path) < 2 * math.sqrt(2) + 0.1


def test_vis_graph_path_gen_blocked():
    world = vertexnav.world.ProposedWorld(vertices=[(0, -2), (0, 1)],
                                          walls=[[(0, -2), (0, 1)]])
    vgraph = vertexnav.planning.VisibilityGraph(world)
    bf = vertexnav.planning.Frontier(geometry.LineString([(0, 1), (0, 5)]))

    path = vgraph.get_shortest_path(start_point=(-1, 0),
                                    end_point=(1, 0),
                                    blocked_frontiers=[bf])

    def path_length(path):
        return sum([
            math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            for p1, p2 in zip(path[:-1], path[1:])
        ])

    assert path[0] == (-1, 0)
    assert path[-1] == (1, 0)
    assert path_length(path) >= 2 * math.sqrt(1.0**2 + 2.0**2)
    assert path_length(path) < 2 * math.sqrt(1.0**2 + 2.0**2) + 0.1


def test_vis_graph_robot_motion():
    # Generate a simple world
    obstacles, boundary = vertexnav.environments.simulated.build_hallway_from_path(
        path=([0, 0], [100, 0], [100, 100]), width=20)
    world = vertexnav.world.World(obstacles=obstacles, boundary=boundary)

    inflation_rad = 4.0

    Pose = vertexnav.Pose

    robot = vertexnav.robot.Turtlebot_Robot(Pose(0, 0, 0))
    goal = Pose(100, 100)

    nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()
    prev_pose = None
    for _ in range(200):
        obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            world.get_vertices_for_pose(robot.pose),
            robot.pose,
            do_add_noise=True,
            cov_rt=[[1, 0], [0, 0.5]])
        if prev_pose is None:
            nvg.add_observation(obs, robot.pose)
        else:
            odom = vertexnav.Pose.get_odom(p_new=robot.pose, p_old=prev_pose)
            nvg.add_observation(obs, odom=odom)

        prev_pose = robot.pose
        proposed_world = nvg.get_proposed_world()

        visibility_graph = vertexnav.planning.VisibilityGraph(
            proposed_world, inflation_rad=0.8 * inflation_rad)
        shortest_path = visibility_graph.get_shortest_path(
            start_point=(robot.pose.x, robot.pose.y),
            end_point=(goal.x, goal.y))
        robot.move(goal, shortest_path, inflation_rad=inflation_rad)

    assert abs(robot.pose.x - 100) < 2
    assert abs(robot.pose.y - 100) < 2


def test_h_value_computation():
    world = vertexnav.world.ProposedWorld(vertices=[(0, -2), (0, 1)],
                                          walls=[[(0, -2), (0, 1)]])

    segment1 = ((-1, 0), (2, 0))
    segment2 = ((2, 0), (-1, 0))
    segment3 = ((-1, 0), (0, 2))
    segment4 = ((0, 2), (2, 0))

    h1 = vertexnav.planning.compute_h_value_for_segment(
        segment1, world.vertices)
    h2 = vertexnav.planning.compute_h_value_for_segment(
        segment2, world.vertices)
    h3 = vertexnav.planning.compute_h_value_for_segment(
        segment3, world.vertices)
    h4 = vertexnav.planning.compute_h_value_for_segment(
        segment4, world.vertices)
    print(h1)
    print((h1 + h2))
    print((h3 + h4))
    print((h3 + h4 + h2))
    assert np.all(h1 + h2 == complex(0, 0))
    assert np.any(h3 + h4 + h2 != complex(0, 0))
