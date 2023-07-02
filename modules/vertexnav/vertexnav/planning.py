import math
import cmath
import numpy as np
import random
import scipy
import shapely


class Frontier():
    """ Defines a single frontier between vertices in polygon

    Attributes:
        linestring: line connecting vertices
        centroid (tuple): position of centroid of linestring
    """
    def __init__(self, linestring):
        self.linestring = linestring
        if isinstance(self.linestring, shapely.geometry.GeometryCollection):
            self.centroid = shapely.geometry.Point(10000, 10000)
        else:
            self.centroid = self.linestring.centroid

    def __hash__(self):
        return hash(str(self.linestring))


def get_boundary_vertices_from_poly(known_poly, world, inflation_rad):
    """Generates a list of vertices on the boundary of the known poly."""
    boundary = known_poly.boundary

    vpoints = [
        shapely.geometry.Point(vert[0], vert[1]) for vert in world.vertices
    ]
    boundary_verts = [
        vert for (vert, vpoint) in zip(world.vertices, vpoints)
        if boundary.intersects(
            vpoint.buffer(
                inflation_rad / 10, resolution=8, cap_style=3, join_style=2))
    ]

    return boundary_verts


def compute_h_value_for_segment(segment, vertices):
    """Generates the 'h_value' (homotopy) for a segment of a path"""
    z1 = segment[0]
    z2 = segment[1]
    h = np.zeros(len(vertices), dtype=np.complex_)
    for ii, vertex in enumerate(vertices):
        ks = list(range(-2, 3))
        c1 = complex(z1[0] - vertex[0], z1[1] - vertex[1])
        c2 = complex(z2[0] - vertex[0], z2[1] - vertex[1])
        argdiff = cmath.phase(c1) - cmath.phase(c2)
        arglist = [argdiff + 2 * k * math.pi for k in ks]
        absargmin = np.argmin([abs(arg) for arg in arglist])
        absmin = arglist[absargmin]
        h[ii] = complex(math.log(abs(c1)) - math.log(abs(c2)), absmin)
    return h


def compute_frontiers_from_poly(known_poly, world, inflation_rad):
    """Generates a list of frontiers from a known poly and a proposed world."""
    try:
        boundary = known_poly.boundary
    except:  # noqa
        return []

    # Compute the boundary where walls do not exist:
    for obs in world.obstacles:
        boundary = boundary.difference(
            obs.buffer(inflation_rad / 10,
                       resolution=8,
                       cap_style=3,
                       join_style=2))

    for vert in world.vertices:
        vpoint = shapely.geometry.Point(vert[0], vert[1])
        boundary = boundary.difference(
            vpoint.buffer(inflation_rad / 10,
                          resolution=8,
                          cap_style=3,
                          join_style=2))

    if isinstance(boundary, shapely.geometry.MultiLineString):
        frontiers = [Frontier(ls) for ls in boundary]
    else:
        frontiers = [Frontier(boundary)]

    def man_dist_from_point(f, point):
        f_cent = f.centroid
        return abs(f_cent.x - point[0]) + abs(f_cent.y - point[1])

    frontiers = sorted(frontiers,
                       key=lambda f: man_dist_from_point(f, [-3170, 1280]))

    return frontiers


def compute_path_length(path):
    """Return length of a path"""
    return sum([
        math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        for p1, p2 in zip(path[:-1], path[1:])
    ])


def multiagent_select_frontiers_greedy(robots,
                                       frontiers,
                                       visibility_graph,
                                       do_explore,
                                       known_space_poly,
                                       goal=None,
                                       penalty=10000,
                                       nearby_clutter_fn=None,
                                       cl_inflation_rad=None):
    """Helper function for planning with multiple agents.

    Computes the cost of planning through each frontier, either to a goal or to
    explore, for each agent and uses the Hungarian Matching Algorithm to select
    the frontiers that minimize net cost. If fewer frontiers than agents exist,
    extra agents will still navigate to their nearest frontier.
    """

    if not do_explore and goal is None:
        raise ValueError('If not exploring, goal must be provided')
    elif do_explore and known_space_poly is None:
        raise ValueError('If exploring, known_space_poly must be provided')

    if not frontiers:
        return []

    # Compute the costs
    all_paths_dict = dict()

    def get_path_cost(robot, blocked_frontiers, do_get_clutter=False):
        if nearby_clutter_fn and do_get_clutter:
            nearby_clutter = nearby_clutter_fn(robot)
        else:
            nearby_clutter = None

        # Compute cost
        if do_explore:
            path, cost = visibility_graph.get_shortest_path(
                start_point=(robot.pose.x, robot.pose.y),
                known_space_poly=known_space_poly,
                blocked_frontiers=blocked_frontiers,
                do_return_cost=True,
                nearby_clutter=nearby_clutter,
                cl_inflation_rad=cl_inflation_rad)
        else:
            path, cost = visibility_graph.get_shortest_path(
                start_point=(robot.pose.x, robot.pose.y),
                end_point=(goal.x, goal.y),
                blocked_frontiers=blocked_frontiers,
                do_return_cost=True,
                nearby_clutter=nearby_clutter,
                cl_inflation_rad=cl_inflation_rad)

        return path, cost

    def get_frontier_for_plan(plan):
        plan_ls = shapely.geometry.LineString(plan)
        for f in frontiers:
            if f.linestring.intersects(plan_ls):
                return f

    # First get the costs for all the robot plans
    rf_data = dict()
    included_frontiers = set()
    prev_len = 0
    while len(included_frontiers) < min(len(robots), len(frontiers)):
        tmp_frontiers = set()
        print((len(robots), len(frontiers), len(included_frontiers)))
        for rind, robot in enumerate(robots):
            path, cost = get_path_cost(robot, included_frontiers)
            local_frontier = get_frontier_for_plan(path)
            rf_data[(rind, hash(local_frontier))] = (path, cost)
            tmp_frontiers.add(local_frontier)

        included_frontiers.update(tmp_frontiers)
        if prev_len == len(included_frontiers):
            break
        else:
            prev_len = len(included_frontiers)

    # Now run the matching algorithm on the result
    included_frontiers = list(included_frontiers)
    mul = int(math.ceil(1.0 * len(robots) / len(included_frontiers)))
    nf = len(included_frontiers)
    cost_mat = 10000 * penalty * np.ones([len(robots), mul * nf])
    for rind, robot in enumerate(robots):
        for find, frontier in enumerate(included_frontiers):
            key = (rind, hash(frontier))
            if key in list(rf_data.keys()):
                path, cost = rf_data[key]

                all_paths_dict[(rind, find)] = path

                for ii in range(mul):
                    cost_mat[rind, find + ii * nf] = cost + ii * penalty

    print(cost_mat)

    finds, rinds = scipy.optimize.linear_sum_assignment(cost_mat.T)

    def get_remaining_frontiers(selected_frontier):
        if selected_frontier is None:
            return []
        else:
            return set(frontiers).difference({selected_frontier})

    return [
        (included_frontiers[fi % nf],
         get_path_cost(robots[ri],
                       get_remaining_frontiers(included_frontiers[fi % nf]),
                       do_get_clutter=True)[0], robots[ri])
        for ri, fi in zip(rinds, finds)
    ]


def multiagent_select_frontiers(robots,
                                frontiers,
                                visibility_graph,
                                do_explore,
                                known_space_poly,
                                goal=None,
                                penalty=10000,
                                nearby_clutter_fn=None,
                                cl_inflation_rad=None):
    """Helper function for planning with multiple agents.

    Computes the cost of planning through each frontier, either to a goal or to
    explore, for each agent and uses the Hungarian Matching Algorithm to select
    the frontiers that minimize net cost. If fewer frontiers than agents exist,
    extra agents will still navigate to their nearest frontier.
    """

    if not do_explore and goal is None:
        raise ValueError('If not exploring, goal must be provided')
    elif do_explore and known_space_poly is None:
        raise ValueError('If exploring, known_space_poly must be provided')

    if not frontiers:
        return []

    # Compute the costs
    mul = int(math.ceil(1.0 * len(robots) / len(frontiers)))
    nf = len(frontiers)
    cost_mat = np.zeros([len(robots), mul * nf])
    all_paths_dict = dict()

    def get_path_cost(robot, frontier, do_get_clutter=False):
        if nearby_clutter_fn and do_get_clutter:
            nearby_clutter = nearby_clutter_fn(robot)
        else:
            nearby_clutter = None

        # Compute cost
        if do_explore:
            path, cost = visibility_graph.get_shortest_path(
                start_point=(robot.pose.x, robot.pose.y),
                known_space_poly=known_space_poly,
                blocked_frontiers=set(frontiers).difference({frontier}),
                do_return_cost=True,
                nearby_clutter=nearby_clutter,
                cl_inflation_rad=cl_inflation_rad)
        else:
            path, cost = visibility_graph.get_shortest_path(
                start_point=(robot.pose.x, robot.pose.y),
                end_point=(goal.x, goal.y),
                blocked_frontiers=set(frontiers).difference({frontier}),
                do_return_cost=True,
                nearby_clutter=nearby_clutter,
                cl_inflation_rad=cl_inflation_rad)

        return path, cost

    for rind, robot in enumerate(robots):
        for find, frontier in enumerate(frontiers):
            path, cost = get_path_cost(robot, frontier)

            all_paths_dict[(rind, find)] = path

            for ii in range(mul):
                cost_mat[rind, find + ii * nf] = cost + ii * penalty

    finds, rinds = scipy.optimize.linear_sum_assignment(cost_mat.T)

    return [(frontiers[fi % nf],
             get_path_cost(robots[ri], frontiers[fi],
                           do_get_clutter=True)[0], robots[ri])
            for ri, fi in zip(rinds, finds)]

    # return [(frontiers[fi % nf], all_paths_dict[(ri, fi % nf)], robots[ri])
    #         for ri, fi in zip(rinds, finds)]


class VisibilityGraph(object):
    """Defines the visibility graph for a proposed world

    Arguments:
        proposed_world (World object): World defining vertices and walls
        inflation_rad (float): How much to inflate obstacles for v-graph
    """
    def __init__(self, proposed_world, inflation_rad=0.01):
        self._world = proposed_world
        self._vertices, self._inflation_obstacles = \
            self._world.get_inflated_vertices(inflation_rad)
        self.inflation_rad = inflation_rad

    def get_shortest_path(self,
                          start_point,
                          end_point=None,
                          known_space_poly=None,
                          blocked_frontiers={},
                          do_return_cost=False,
                          nearby_clutter=None,
                          cl_inflation_rad=None):
        """Return shortest path between two points through visibility graph. If
        end point is not specified, sample one at random. Optionally return the
        cost of the optimal path.
        """

        cl_inflated_verts = []
        cl_inflated_obstacles = []
        if nearby_clutter:
            for cl in nearby_clutter:
                print(("  CLUTTER: {}".format(cl)))
                num_angles = 6
                poly_verts = []
                for th in np.linspace(0,
                                      2 * math.pi,
                                      num=num_angles,
                                      endpoint=False):
                    cl_inflated_verts.append(
                        (cl.x + cl_inflation_rad * math.cos(th),
                         cl.y + cl_inflation_rad * math.sin(th)))
                    poly_verts.append(
                        (cl.x + 0.95 * cl_inflation_rad * math.cos(th),
                         cl.y + 0.95 * cl_inflation_rad * math.sin(th)))

                cl_inflated_obstacles.append(
                    shapely.geometry.Polygon(poly_verts))

        mdist = 100000000.0
        # known_space_poly = known_space_poly.buffer(
        #     self.inflation_rad/10)

        if end_point is not None:
            verts = self._vertices + cl_inflated_verts + [
                start_point, end_point
            ]
            start_ind = len(verts) - 2
            end_inds = [len(verts) - 1]
        elif known_space_poly is not None:

            extra_verts = []
            if isinstance(known_space_poly, shapely.geometry.Polygon):

                def get_random_point(polygon):
                    """Helper function: returns random point in polygon"""
                    minx, miny, maxx, maxy = polygon.bounds
                    for _ in range(1000):
                        point = shapely.geometry.Point(
                            random.uniform(minx, maxx),
                            random.uniform(miny, maxy))
                        if polygon.contains(point):
                            return point

                    return None

                for poly in list(known_space_poly.interiors):
                    point = get_random_point(shapely.geometry.Polygon(poly))
                    point = shapely.geometry.Polygon(poly).centroid
                    if point:
                        extra_verts.append((point.x, point.y))

            verts = self._vertices + cl_inflated_verts + extra_verts + [
                start_point
            ]
            start_ind = len(verts) - 1
            end_inds = []
            if not len(verts):
                if do_return_cost:
                    return ([start_point, start_point], mdist)
                else:
                    return [start_point, start_point]

            for ind, v in enumerate(verts):
                if v in cl_inflated_verts:
                    continue

                vp = shapely.geometry.Point(v)
                try:
                    does_contain = known_space_poly.contains(vp)
                except:  # noqa
                    does_contain = False

                if not does_contain and ind is not start_ind:
                    for bf in blocked_frontiers:
                        if bf and bf.linestring.buffer(
                                self.inflation_rad / 10).contains(vp):
                            break
                    else:
                        end_inds.append(ind)

            if not len(end_inds):
                if do_return_cost:
                    return ([start_point, start_point], mdist)
                else:
                    return [start_point, start_point]
        else:
            raise ValueError("Either 'end_point' or 'known_space_poly' " +
                             "must be provided to 'get_shortest_path'.")

        def get_neighbors(ind):
            """Helper: return neighbors in visibility graph"""
            possible_edges = [(ind, ii) for ii in range(len(verts))
                              if not ii == ind]
            return [
                e[1] for e in possible_edges if self._world.is_covisible(
                    verts[e[0]], verts[e[1]], self._inflation_obstacles)
            ]

        def does_intersect_block(p1, p2):
            """Helper: returns 1 if path passes through blocked frontier"""
            ls = shapely.geometry.LineString((p1, p2))
            for bf in blocked_frontiers:
                if bf and ls.intersects(bf.linestring):
                    return 1.0

            for bcl in cl_inflated_obstacles:
                if ls.intersects(bcl):
                    return 1.0

            return 0.0

        start_neighbors = get_neighbors(start_ind)

        def get_neighbor_dists(ind, neighbors):
            def dist(p1, p2):
                return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

            def get_explore_cost(inda, indb):
                o_ind = None
                if inda in start_neighbors and indb == start_ind:
                    o_ind = inda
                elif indb in start_neighbors and inda == start_ind:
                    o_ind = indb

                if o_ind is None:
                    return 0.0

                if o_ind in end_inds:  # and start_ind not in end_inds:

                    d = dist(verts[o_ind], verts[start_ind])
                    s = 0.999 - 0.00 * max(1 - 0.5 * d / self.inflation_rad, 0)
                    return 0.0 - s * d
                else:
                    return 0.0

            return [
                (ni,
                 dist(verts[ind], verts[ni]) + 0 * get_explore_cost(ind, ni) +
                 100000 * does_intersect_block(verts[ind], verts[ni]))
                for ni in neighbors
            ]

        # Initializations
        is_visited = np.zeros(len(verts))
        upstream_node = -1 * np.ones(len(verts), dtype=int)
        node_dists = np.ones(len(verts)) * mdist

        # Dijkstra's algorithm
        current_ind = start_ind
        node_dists[current_ind] = 0
        stored_neighbors = {}

        while current_ind not in end_inds:
            is_visited[current_ind] = 1
            current_dist = node_dists[current_ind]
            neighbors = get_neighbors(current_ind)
            stored_neighbors[current_ind] = neighbors
            ndists = get_neighbor_dists(current_ind, neighbors)

            # Update neighbors
            for ni, dist in ndists:
                new_dist = current_dist + dist
                if new_dist < node_dists[ni]:
                    node_dists[ni] = new_dist
                    upstream_node[ni] = current_ind

            # Pick new node
            current_ind = np.argmin(node_dists * (1 - is_visited) + 2 * mdist *
                                    (is_visited))

        # Get shortest path from graph
        path_inds = [current_ind]
        if node_dists[current_ind] >= mdist:
            print("Greater than mdist")
            if do_return_cost:
                return ([start_point, start_point], mdist)
            else:
                return [start_point, start_point]

        while node_dists[path_inds[-1]] > 0.0001:
            new_ind = int(upstream_node[path_inds[-1]])
            path_inds.append(new_ind)

        if len(path_inds) == 1:
            path_inds.append(start_ind)

        if do_return_cost:
            return ([verts[ii]
                     for ii in path_inds[::-1]], node_dists[path_inds[0]])
        else:
            return [verts[ii] for ii in path_inds[::-1]]
