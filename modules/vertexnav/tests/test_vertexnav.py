import math
import matplotlib.pyplot as plt
import pytest

import vertexnav
from vertexnav.world import World
from vertexnav_accel import Pose

import shapely
from shapely import geometry


def get_world_square():
    # A square
    square_poly = geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    return World(obstacles=[square_poly])


def get_world_two_squares():
    # A square
    square_poly_1 = geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    square_poly_2 = geometry.Polygon([(3, 1.6), (3, 2.5), (4, 2.5), (4, 1.6)])

    return World(obstacles=[square_poly_1, square_poly_2])


def get_world_overlapping_shapes():
    # A square
    square_poly_1 = geometry.Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    square_poly_2 = geometry.Polygon([(1.25, 1.5), (3, 2.5), (4, 2.5),
                                      (4, 1.5)])

    return World(obstacles=[square_poly_1.union(square_poly_2)])


def test_world_square_corner_detection():
    """Simple world with only a single square in it."""
    square_world = get_world_square()

    assert (len(square_world.get_all_vertices()) == 4)

    assert (len(
        square_world.get_visible_vertices_for_pose(
            robot_pose=Pose(x=0.0, y=0.0))) == 3)

    assert (len(
        square_world.get_visible_vertices_for_pose(
            robot_pose=Pose(x=1.5, y=0.0))) == 2)

    # If inside square, no vertices are expected to be visible (obstacle)
    assert (len(
        square_world.get_visible_vertices_for_pose(
            robot_pose=Pose(x=1.5, y=1.5))) == 4)


def test_world_square_gap_detection():
    """Simple world with only a single square in it."""
    square_world = get_world_square()

    def count_gaps(vertices):
        return sum([v[1] == 'l' or v[1] == 'r' for v in vertices])

    side_vertices = square_world.get_vertices_for_pose(
        robot_pose=Pose(x=0.0, y=0.0))
    assert (count_gaps(side_vertices) == 2)
    assert (sum([v[1] == 'c' for v in side_vertices]) == 1)
    assert (sum([v[1] == 'l' for v in side_vertices]) == 1)
    assert (sum([v[1] == 'r' for v in side_vertices]) == 1)

    side_vertices = square_world.get_vertices_for_pose(
        robot_pose=Pose(x=1.5, y=0.0))
    assert (count_gaps(side_vertices) == 2)
    assert (sum([v[1] == 'l' for v in side_vertices]) == 1)
    assert (sum([v[1] == 'r' for v in side_vertices]) == 1)

    inside_vertices = square_world.get_vertices_for_pose(
        robot_pose=Pose(x=1.5, y=1.5))
    assert len(inside_vertices) == 4
    assert (sum([v[1] == 'c' for v in inside_vertices]) == 4)


def test_world_two_squares_corner_detection():
    """Simple world with only a single square in it."""
    square_world = get_world_two_squares()

    assert (len(square_world.get_all_vertices()) == 8)

    assert (len(
        square_world.get_visible_vertices_for_pose(
            robot_pose=Pose(x=0.0, y=0.0))) == 4)

    assert (len(
        square_world.get_visible_vertices_for_pose(
            robot_pose=Pose(x=1.5, y=0.0))) == 5)

    assert (len(
        square_world.get_visible_vertices_for_pose(
            robot_pose=Pose(x=1.5, y=1.5))) == 4)


def test_world_signed_distance():
    """Confirms that the signed distance function works as expected
    for a simple square world."""

    world = get_world_square()

    assert world.get_signed_dist(Pose(x=0, y=0)) == pytest.approx(math.sqrt(2))
    assert world.get_signed_dist(Pose(x=1, y=1)) == pytest.approx(0.0)
    assert world.get_signed_dist(Pose(x=-10, y=1)) == pytest.approx(11.0)
    assert world.get_signed_dist(Pose(x=1.5, y=1.5)) == pytest.approx(-0.5)


def test_proposed_world_get_visible():
    """Show that I can initialize a "ProposedWorld" with a set of walls."""

    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    walls = [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)),
             ((0, 1), (0, 0))]

    proposed_world = vertexnav.world.ProposedWorld(vertices=vertices,
                                                   walls=walls)

    verts = proposed_world.get_vertices_for_pose(Pose(x=0.5, y=0.5))
    assert len(verts) == 4
    assert len([v for v in verts if v[1] == 'c']) == 4

    verts = proposed_world.get_vertices_for_pose(Pose(x=-1.0, y=-1.0))
    num_c = len([v for v in verts if v[1] == 'c'])
    num_l = len([v for v in verts if v[1] == 'l'])
    num_r = len([v for v in verts if v[1] == 'r'])
    assert len(verts) == 3
    assert num_l == 1
    assert num_r == 1
    assert num_c == 1

    verts = proposed_world.get_vertices_for_pose(Pose(x=-0.5, y=0.5))
    num_l = len([v for v in verts if v[1] == 'l'])
    num_r = len([v for v in verts if v[1] == 'r'])
    assert len(verts) == 2
    assert num_l == 1
    assert num_r == 1


def test_proposed_world_get_visible_max_range():
    """Show that I can initialize a "ProposedWorld" with a set of walls."""

    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    walls = [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)),
             ((0, 1), (0, 0))]

    proposed_world = vertexnav.world.ProposedWorld(vertices=vertices,
                                                   walls=walls)

    verts = proposed_world.get_vertices_for_pose(Pose(x=0.5, y=0.5),
                                                 max_range=0.1)
    assert len(verts) == 0

    verts = proposed_world.get_vertices_for_pose(Pose(x=0.5, y=0.5),
                                                 max_range=1.0)
    assert len(verts) == 4

    # When the max range is limited, the vertex class should also change
    verts = proposed_world.get_vertices_for_pose(Pose(x=-1.0, y=-1.0),
                                                 max_range=math.sqrt(2.5))
    assert len(verts) == 1
    assert len([v for v in verts if v[1] == 'p']) == 1

    verts = proposed_world.get_vertices_for_pose(Pose(x=-1.0, y=-1.0),
                                                 max_range=math.sqrt(8))
    assert len(verts) == 3
    assert len([v for v in verts if v[1] == 'p']) == 0


def test_hallway_world_clutter():
    """Confirm that clutter generation respects signed distance."""
    num_elements = 100
    min_signed_dist = -0.5
    max_signed_dist = 1.0
    world = vertexnav.environments.simulated.HallwayWorld(
        num_clutter_elements=num_elements,
        max_clutter_signed_distance=max_signed_dist,
        min_clutter_signed_distance=min_signed_dist)

    assert len(world.clutter_element_poses) == num_elements

    for pose in world.clutter_element_poses:
        signed_dist = world.get_signed_dist(pose)
        assert signed_dist >= min_signed_dist
        assert signed_dist <= max_signed_dist


def test_get_visibility_polygon_world():
    pytest.xfail("Plotting for debugging purposes")

    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    walls = [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)),
             ((0, 1), (0, 0))]

    vertices = [(0, 0), (3, 0), (3, 3), (0, 5), (2.5, 6), (1, 2), (2, 4),
                (1, 4)]

    walls = [((0, 0), (3, 0)), ((3, 0), (3, 3)), ((0, 0), (0, 5)),
             ((0, 5), (2.5, 6)), ((1, 2), (2, 4)), ((1, 2), (1, 4)),
             ((1, 4), (2, 4))]

    proposed_world = vertexnav.world.ProposedWorld(vertices=vertices,
                                                   walls=walls)

    # from matplotlib.patches import Polygon
    # from matplotlib.collections import PatchCollection
    # import shapely

    # patches = []
    # colors = []

    # pose = Pose(0.48, 1)
    # color = [1, 0, 0]
    # poly_points = vertexnav.noisy.get_visibility_polygon(
    #     pose, proposed_world, radius=10, do_cut_with_world=True)
    # poly_a = shapely.geometry.Polygon(poly_points)
    # plt.plot(pose.x, pose.y, 'o', color=color)
    # vertexnav.plotting.plot_polygon(plt.gca(), poly_a,
    #                              color=color, alpha=0.1)
    # poly_points = vertexnav.noisy.get_visibility_polygon(
    #     pose, proposed_world, radius=-1, do_cut_with_world=True)
    # poly_a = shapely.geometry.Polygon(poly_points)
    # plt.plot(pose.x, pose.y, 'o', color=color)
    # vertexnav.plotting.plot_polygon(plt.gca(), poly_a,
    #                              color=color, alpha=0.1)

    # pose = Pose(1.2, 4.5)
    # color = [0, 0, 1]
    # poly_points = vertexnav.noisy.get_visibility_polygon(
    #     pose, proposed_world, radius=10, do_cut_with_world=True)
    # poly_b = shapely.geometry.Polygon(poly_points)
    # plt.plot(pose.x, pose.y, 'o', color=color)
    # vertexnav.plotting.plot_polygon(plt.gca(), poly_b,
    #                              color=color, alpha=0.1)
    # poly_points = vertexnav.noisy.get_visibility_polygon(
    #     pose, proposed_world, radius=-1, do_cut_with_world=True)
    # poly_b = shapely.geometry.Polygon(poly_points)
    # plt.plot(pose.x, pose.y, 'o', color=color)
    # vertexnav.plotting.plot_polygon(plt.gca(), poly_b,
    #                              color=color, alpha=0.1)

    pose = Pose(5.45, 1.3)
    color = [0, 1, 0]
    poly_points = vertexnav.noisy.get_visibility_polygon(pose,
                                                         proposed_world,
                                                         radius=10,
                                                         is_conservative=True)
    poly_b = shapely.geometry.Polygon(poly_points)
    plt.plot(pose.x, pose.y, 'o', color=color)
    vertexnav.plotting.plot_polygon(plt.gca(), poly_b, color=color, alpha=0.1)
    poly_points = vertexnav.noisy.get_visibility_polygon(pose,
                                                         proposed_world,
                                                         radius=10,
                                                         is_conservative=False)
    poly_b = shapely.geometry.Polygon(poly_points)
    plt.plot(pose.x, pose.y, 'o', color=color)
    vertexnav.plotting.plot_polygon(plt.gca(), poly_b, color=color, alpha=0.1)

    # poly = shapely.ops.cascaded_union([poly_a, poly_b])
    # vertexnav.plotting.plot_polygon(plt.gca(), poly,
    #                              color=[0.0, 0.0, 0.0], alpha=0.2)
    vertexnav.plotting.plot_proposed_world(plt.gca(), proposed_world)

    plt.show()

    assert False
