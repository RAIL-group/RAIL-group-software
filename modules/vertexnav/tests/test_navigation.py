import vertexnav
import pytest
import shapely


def test_get_inflated_vertices():

    # Simple world, with only 1 line, we should expect 4 vertices
    verts = [(0, 0), (1, 0), (1, 1)]
    world = vertexnav.world.ProposedWorld(vertices=verts,
                                          walls=[[(0, 0), (1, 0)]])

    # Solitary vert
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0)], walls=[])
    inf_verts, _ = world.get_inflated_vertices(inflation_rad=0.1)
    assert len(inf_verts) > 1

    # One vert, single wall
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0)],
                                          walls=[[(0, 0), (1, 0)]])
    inf_verts, _ = world.get_inflated_vertices(inflation_rad=0.1)
    assert len(inf_verts) > 2

    # Two verts, single wall
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0), (1, 0)],
                                          walls=[[(0, 0), (1, 0)]])
    inf_verts, _ = world.get_inflated_vertices(inflation_rad=0.1)
    assert len([v for v in inf_verts if v[0] < 0]) >= 2
    assert len([v for v in inf_verts if v[0] > 1]) >= 2
    assert len([v for v in inf_verts if v[1] < 0]) >= 2
    assert len([v for v in inf_verts if v[1] > 0]) >= 2

    # Single vert with 3 walls connected to it
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0)],
                                          walls=[[(0, 0), (1, 0)],
                                                 [(0, 0), (-1, 0)],
                                                 [(0, 0), (0, 1)]])
    inf_verts, _ = world.get_inflated_vertices(inflation_rad=0.1)
    assert len(inf_verts) == 3
    assert len([v for v in inf_verts if v[0] > 0.01]) == 1
    assert len([v for v in inf_verts if v[0] < -0.01]) == 1
    assert len([v for v in inf_verts if v[1] < 0]) == 1
    assert len([v for v in inf_verts if v[1] > 0]) == 2

    # 3 verts joined at a 4th
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0), (1, 0.4),
                                                    (-1, 0.4), (0, 1)],
                                          walls=[[(0, 0), (1, 0.4)],
                                                 [(0, 0), (-1, 0.4)],
                                                 [(0, 0), (0, 1)]])
    inf_verts, _ = world.get_inflated_vertices(inflation_rad=0.1)
    assert len(inf_verts) >= 9


def test_get_inflated_verts_and_visibility_one_vert():
    """Confirm that the inflated vertices and visibility edges conform to our expectation."""

    # One vert, single wall
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0)],
                                          walls=[[(0, 0), (1, 0)]])

    inf_verts, inf_obs = world.get_inflated_vertices(inflation_rad=0.01)
    vis_edges = world.get_visibility_edges_from_verts(inf_verts, inf_obs)

    assert len(world.vertices) == 1
    assert len(inf_verts) > 1
    assert len(inf_obs) == 1
    assert len(vis_edges) == len(inf_verts) - 1

    world_b = vertexnav.world.ProposedWorld(vertices=[(0, 0)],
                                            walls=[[(0, 0), (1, 1)]])

    inf_verts_b, inf_obs_b = world_b.get_inflated_vertices(inflation_rad=0.01)
    vis_edges_b = world_b.get_visibility_edges_from_verts(
        inf_verts_b, inf_obs_b)

    assert len(world.vertices) == 1
    assert len(inf_verts_b) > 1
    assert len(inf_obs_b) == 1
    assert len(vis_edges_b) == len(inf_verts_b) - 1
    assert not inf_verts == inf_verts_b
    assert not vis_edges == vis_edges_b

    world_c = vertexnav.world.ProposedWorld(vertices=[(1, 0)],
                                            walls=[[(0, 0), (1, 0)]])

    inf_verts_c, inf_obs_c = world_c.get_inflated_vertices(inflation_rad=0.01)
    vis_edges_c = world_c.get_visibility_edges_from_verts(
        inf_verts_c, inf_obs_c)

    assert len(world.vertices) == 1
    assert len(inf_verts_c) > 1
    assert len(inf_obs_c) == 1
    assert len(vis_edges_c) == len(inf_verts_c) - 1
    assert not inf_verts == inf_verts_c
    assert not vis_edges == vis_edges_c


def test_get_inflated_verts_and_visibility_two_verts():
    """Confirm that the inflated vertices and visibility edges conform to our expectation."""

    # One vert, single wall
    world = vertexnav.world.ProposedWorld(vertices=[(0, 0)],
                                          walls=[[(0, 0), (1, 0)]])

    inf_verts, inf_obs = world.get_inflated_vertices(inflation_rad=0.01)
    vis_edges = world.get_visibility_edges_from_verts(inf_verts, inf_obs)

    assert len(world.vertices) == 1
    assert len(inf_verts) > 1
    assert len(inf_obs) == 1
    assert len(vis_edges) == len(inf_verts) - 1

    # One vert, single wall
    world_b = vertexnav.world.ProposedWorld(vertices=[(0, 0), (1, 0)],
                                            walls=[[(0, 0), (1, 0)]])

    inf_verts_b, inf_obs_b = world_b.get_inflated_vertices(inflation_rad=0.01)
    vis_edges_b = world_b.get_visibility_edges_from_verts(
        inf_verts_b, inf_obs_b)

    assert len(world.vertices) == 1
    assert len(inf_verts_b) > 2
    assert len(inf_obs_b) == 2

    assert len(world_b.vertices) == 2
    assert len(inf_verts_b) == 2 * len(inf_verts)
    assert len(inf_obs_b) == 2
    assert len(vis_edges_b) > 2 * len(vis_edges)


@pytest.mark.skip("Test too succeptable to numerical noise.")
def test_get_h_obs_missing_wall():
    """Confirm that the hypothesized observation does not contain points outside the
    visibility region. Test disabled: Note that this functionality is still
    generally expected to succedd in the wild. However, using two different
    "proposed world" instances is much more challenging when computing the
    hypothetical observations because the C++ code is more succeptable to
    numerical noise."""
    underlying_world = vertexnav.world.ProposedWorld(vertices=[(0, 0), (0, 1),
                                                               (1, 1), (1, 0)],
                                                     walls=[[(0, 0), (1, 0)],
                                                            [(1, 0), (1, 1)],
                                                            [(1, 1), (0, 1)],
                                                            [(0, 1), (0, 0)]])
    missing_wall_world = vertexnav.world.ProposedWorld(vertices=[
        (0, 0), (0, 1), (1, 1), (1, 0)], walls=[[(1, 0), (1, 1)],
                                                [(1, 1), (0, 1)],
                                                [(0, 1),
                                                (0, 0)]])
    r_pose = vertexnav.Pose(-0.5, -1.0)

    vertices = underlying_world.get_vertices_for_pose(r_pose)
    observation = vertexnav.noisy.convert_world_obs_to_noisy_detection(
        vertices, r_pose)
    naive_h_vertices = missing_wall_world.get_vertices_for_pose(r_pose)
    naive_h_observation = vertexnav.noisy.convert_world_obs_to_noisy_detection(
        naive_h_vertices, r_pose)
    h_observation = vertexnav.noisy.compute_hypothetical_observation(
        missing_wall_world, r_pose, observation)

    assert len(underlying_world.vertices) == 4
    assert len(missing_wall_world.vertices) == 4
    assert len(observation) == 3
    assert len(naive_h_vertices) == 4
    assert len(h_observation) == 3

    def assert_in_obs(val):
        for obs in h_observation:
            if obs.position == pytest.approx(val):
                break
        else:
            raise ValueError(
                "{} does not appear in h_observtation.".format(val))

    assert_in_obs([0, 0])
    assert_in_obs([0, 1])
    assert_in_obs([1, 0])
    with pytest.raises(ValueError):
        assert_in_obs([1, 1])

    # Now confirm that the visibility polygons have the properties we expect
    obs_poly = shapely.geometry.Polygon(
        vertexnav.noisy.compute_conservative_space_from_obs(
            r_pose, observation))
    h_obs_poly = shapely.geometry.Polygon(
        vertexnav.noisy.compute_conservative_space_from_obs(
            r_pose, h_observation))
    naive_h_obs_poly = shapely.geometry.Polygon(
        vertexnav.noisy.compute_conservative_space_from_obs(
            r_pose, naive_h_observation))

    assert obs_poly.area >= h_obs_poly.area
    assert not obs_poly.intersects(shapely.geometry.Point(0.5, 0.25))
    assert not h_obs_poly.intersects(shapely.geometry.Point(0.5, 0.25))
    assert naive_h_obs_poly.intersects(shapely.geometry.Point(0.5, 0.25))


def test_get_h_obs_false_neg():
    """Confirm that the hypothesized observation does not contain points
    outside the visibility region."""
    pytest.xfail("Test does not conform to new C++ API.")

    underlying_world = vertexnav.world.ProposedWorld(vertices=[(0.0, 0.0),
                                                               (0.0, 1.0),
                                                               (1.0, 1.0),
                                                               (1.0, 0.0)],
                                                     walls=[[(0.0, 0.0),
                                                             (1.0, 0.0)],
                                                            [(1.0, 0.0),
                                                             (1.0, 1.0)],
                                                            [(1.0, 1.0),
                                                             (0.0, 1.0)],
                                                            [(0.0, 1.0),
                                                             (0.0, 0.0)]])

    nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()
    for ii in range(10):
        r_pose = vertexnav.Pose(-0.5, -1.0)

        vertices = underlying_world.get_vertices_for_pose(r_pose)
        observation = vertexnav.noisy.convert_world_obs_to_noisy_detection(
            vertices,
            r_pose,
            do_add_noise=False,
            cov_rt=[[0.5, 0.0], [0.0, 0.5]])
        if ii == 0:
            nvg.add_observation(observation, r_pose)
        else:
            nvg.add_observation(observation, odom=vertexnav.Pose(0, 0, 0))

    pw = nvg.get_proposed_world_fast(nvg.topology)
    vertices = pw.vertices + [(0.1, -0.1), (2.0, -0.1)]
    pw_aug = vertexnav.world.ProposedWorld(vertices=vertices,
                                           walls=[[(0.0, 0.0), (1.0, 0.0)],
                                                  [(1.0, 0.0), (1.0, 1.0)],
                                                  [(1.0, 1.0), (0.0, 1.0)],
                                                  [(0.0, 1.0), (0.0, 0.0)]])

    h_observation_correct = vertexnav.noisy.compute_hypothetical_observation(
        pw, r_pose, observation)
    h_observation = vertexnav.noisy.compute_hypothetical_observation(
        pw_aug, r_pose, observation)

    obs_poly = shapely.geometry.Polygon(
        vertexnav.noisy.compute_conservative_space_from_obs(r_pose,
                                                            observation,
                                                            radius=10))
    h_obs_poly = shapely.geometry.Polygon(
        vertexnav.noisy.compute_conservative_space_from_obs(r_pose,
                                                            h_observation,
                                                            radius=10))

    assert len(underlying_world.vertices) == 4
    assert len(pw.vertices) == 3
    assert len(pw_aug.vertices) == 5
    assert len(observation) == 3
    # Only one of the two extra vertices is in the visible region
    assert len(h_observation_correct) == 3
    assert len(h_observation) == 4

    def assert_in_obs(val):
        for obs in h_observation:
            if obs.position == pytest.approx(val):
                break
        else:
            raise ValueError(
                "{} does not appear in h_observtation.".format(val))

    assert_in_obs([0, 0])
    assert_in_obs([0, 1])
    assert_in_obs([1, 0])
    with pytest.raises(ValueError):
        assert_in_obs([1, 1])
    with pytest.raises(ValueError):
        assert_in_obs([-1, -1])

    # Now confirm that the visibility polygons have the properties we expect
    assert obs_poly.area >= h_obs_poly.area


def test_get_h_obs_mixed():
    """This regression test was written to identify a particularly odd scenario in
which additional free space was added that was outside the map despite all
vertices being correctly identified. When the hypothetical observation was
computed, if some points in theactual observation had been disabled, the results
may not be correct."""
    pytest.xfail("Test incomplete; known to fail.")

    nvg = vertexnav.prob_vertex_graph.ProbVertexGraph()

    underlying_world = vertexnav.world.ProposedWorld(vertices=[(3, -3),
                                                               (3, 3)],
                                                     walls=[[(2, 0), (3, -3)],
                                                            [(2, 0), (3, 3)],
                                                            [(3, -3), (3, 3)]])
    incorrect_world = vertexnav.world.ProposedWorld(vertices=[(1, 0), (3, -3),
                                                              (3, 3)],
                                                    walls=[[(1, 0), (3, -3)],
                                                           [(1, 0), (3, 3)],
                                                           [(3, -3), (3, 3)]])
    r_pose = vertexnav.Pose(0.0, 0.0)

    vertices = incorrect_world.get_vertices_for_pose(r_pose)
    observation = vertexnav.noisy.convert_world_obs_to_noisy_detection(
        vertices,
        r_pose,
        do_add_noise=False,
        cov_rt=[[2.0**2, 0], [0, 0.15**2]])
    nvg.add_observation(observation, r_pose)
    h_observation = vertexnav.noisy.compute_hypothetical_observation(
        underlying_world, r_pose, observation)

    assert (len(h_observation) == 3)


def test_clutter_avoid():
    inflation_rad = 1.0
    proposed_world = vertexnav.world.ProposedWorld(walls=[], vertices=[])
    visibility_graph = vertexnav.planning.VisibilityGraph(proposed_world,
                                                          inflation_rad=0.8 *
                                                          inflation_rad)

    start = vertexnav.Pose(0, 0)
    goal = vertexnav.Pose(0, 10)

    _, cost = visibility_graph.get_shortest_path(start_point=(start.x,
                                                              start.y),
                                                 end_point=(goal.x, goal.y),
                                                 do_return_cost=True)

    assert cost == pytest.approx(10.0)

    path, cost = visibility_graph.get_shortest_path(
        start_point=(start.x, start.y),
        end_point=(goal.x, goal.y),
        do_return_cost=True,
        nearby_clutter=[vertexnav.Pose(0.0, 5.0)],
        cl_inflation_rad=inflation_rad)

    assert cost > 10.0
