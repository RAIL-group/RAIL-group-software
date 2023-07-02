import itertools
import math
import numpy as np
import random
from vertexnav_accel import Pose
import shapely
import shapely.prepared
from shapely import geometry, strtree
import vertexnav_accel as gaccel


class WorldBase(object):
    """Abstract class, defines some of the attributes and functions of World"""
    def __init__(self):
        raise NotImplementedError("WorldBase is an abstract class.")

    def get_all_vertices(self):
        """Get all vertices in the map."""
        raise NotImplementedError("abstract method of the WorldBase class.")

    def is_vertex_visible(self, robot_position, corner, inflate_ratio=0.0001):
        """Given a robot position, returns the vertices within line of sight"""
        if not hasattr(self, strtree):
            self.strtree = strtree.STRtree(self.obstacles)

        ir = inflate_ratio
        mr = 1 - inflate_ratio
        pair = np.array([robot_position, corner])
        pair = [mr * pair[0] + ir * pair[1], ir * pair[0] + mr * pair[1]]
        line = geometry.LineString(pair)

        for obs in self.strtree.query(line):
            if self.boundary is not None and obs == self.boundary:
                if not obs.contains(line):
                    return False
            elif line.intersects(obs):
                return False

        return True

    def does_wall_exist(self, pos_a, pos_b):
        """Returns true if wall exists between two points"""
        ws = shapely.geometry.LineString([pos_a, pos_b])
        for obs in self.obstacles:
            if isinstance(obs, shapely.geometry.Polygon):
                if obs.boundary.buffer(1e-3, 1).contains(ws):
                    return True
            elif obs.buffer(1e-3, 1).contains(ws):
                return True
        return False

    def get_nearby_clutter(self, robot_pose, dist_threshold):
        return []

    def get_visible_vertices_for_pose(self,
                                      robot_pose,
                                      filter_fn=None,
                                      bound_points=None):
        """Return gap observation"""
        if self.fast_world is not None:
            if bound_points is not None:
                return self.fast_world.getVisPolyBounded(
                    robot_pose.x, robot_pose.y, bound_points[1:])
            else:
                verts = self.fast_world.getVisPoly(robot_pose.x, robot_pose.y)
                if filter_fn is None:
                    return verts
                else:
                    return [v for v in verts if filter_fn((v[0], v[1]))]

        if filter_fn is None:
            filter_fn = lambda x: True  # noqa: E731

        robot_position = (robot_pose.x, robot_pose.y)
        visible_corners = [
            vertex for vertex in self.get_all_vertices()
            if filter_fn((vertex[0], vertex[1]))
            and self.is_vertex_visible(robot_position, vertex)
        ]
        return visible_corners

    def get_vertex_detection_type_for_vertex_noisy(self, robot_pose, vertex):
        raise NotImplementedError

    def get_vertices_for_pose(self,
                              robot_pose,
                              noisy=False,
                              do_compute_detection_type=True,
                              filter_fn=None,
                              bound_points=None,
                              max_range=None):
        """Returns Corner objects"""

        # Get visible corners
        visible_vertices = self.get_visible_vertices_for_pose(
            robot_pose, filter_fn, bound_points)
        if max_range is not None:
            visible_vertices = [
                v for v in visible_vertices
                if math.sqrt((v[0] - robot_pose.x)**2 +
                             (v[1] - robot_pose.y)**2) < max_range
            ]

        if not do_compute_detection_type:
            return visible_vertices

        def get_angle_rad(pose, position):
            return math.atan2(position[1] - pose.y, position[0] - pose.x)

        visible_vertices = sorted(visible_vertices,
                                  key=lambda v: get_angle_rad(robot_pose, v),
                                  reverse=True)

        def get_delta_angle_rad_ordered(pose, det_L, det_R):
            """Get the difference in angle between detections from
            the perspective of a robot pose"""
            low_angle = get_angle_rad(pose, det_R)
            high_angle = get_angle_rad(pose, det_L)

            if high_angle < low_angle:
                high_angle += 2 * math.pi

            return high_angle - low_angle

        verts = visible_vertices
        sverts = visible_vertices[1:] + visible_vertices[:1]
        ssverts = visible_vertices[2:] + visible_vertices[:2]
        vert_pairs = []
        for det_R, det, det_L in zip(ssverts, sverts, verts):
            if get_delta_angle_rad_ordered(robot_pose, det_L, det) < math.pi \
               and self.does_wall_exist(det_L, det):
                is_left_wall = True
            else:
                is_left_wall = False

            if get_delta_angle_rad_ordered(robot_pose, det, det_R) < math.pi \
               and self.does_wall_exist(det, det_R):
                is_right_wall = True
            else:
                is_right_wall = False

            if is_left_wall and is_right_wall:
                det_char = 'c'
            elif not is_left_wall and is_right_wall:
                det_char = 'l'
            elif is_left_wall and not is_right_wall:
                det_char = 'r'
            elif not is_left_wall and not is_right_wall:
                det_char = 'p'
            else:
                raise ValueError("should not be able to get here.")

            vert_pairs.append((det, det_char))

        return vert_pairs

    def get_visibility_edges(self, inflate_ratio=0.0001):
        """Return edges of visibility graph"""

        # Loop though all pairs of vertices
        vs = self.get_all_vertices()

        def is_points_near(pA, pB):
            return (abs(pA[1] - pB[1]) < 1e-8 and abs(pA[0] - pB[0]) < 1e-8)

        ir = inflate_ratio
        mr = 1 - inflate_ratio

        visibility_lines = []

        for pair in itertools.combinations(vs, 2):
            pair = np.array(pair)
            mid = 0.5 * pair[0] + 0.5 * pair[1]
            if self.get_signed_dist(Pose(mid[0], mid[1])) < 0:
                continue
            pair = [mr * pair[0] + ir * pair[1], ir * pair[0] + mr * pair[1]]
            line = geometry.LineString(pair)
            for obs in self.obstacles:
                intersection = self._compute_intersection(line, obs)
                if isinstance(intersection, geometry.GeometryCollection):
                    pass
                elif isinstance(intersection, geometry.Point):
                    break
                elif isinstance(intersection, geometry.MultiPoint):
                    break
                elif isinstance(intersection, geometry.LineString):
                    break
                else:
                    raise ValueError("Unknown type for 'intersection'")
            else:
                visibility_lines.append(pair)

        return visibility_lines


class World(WorldBase):
    """Stores the shapely polygons that define the world

    Attributes:
        obstacles: list of shapely polygons
        boudary: shapely polygon which defines the outer obstacle of world
        clutter_element_poses: positions of clutter
        known_space_poly: shapely polygon representing known, free space
        area: area of known space polygon
    """
    def __init__(self, obstacles, boundary=None):
        self.obstacles = obstacles
        self.boundary = boundary
        self._internal_obstacles = list(obstacles)
        self.clutter_element_poses = []
        if self.boundary is not None:
            self.obstacles.append(boundary)
            self.known_space_poly = boundary
            for obs in self._internal_obstacles:
                self.known_space_poly = self.known_space_poly.difference(obs)
            self.area = self.known_space_poly.area
        else:
            # FIXME(gjstein): should compute with some bounds
            self.obs_poly = shapely.geometry.Polygon()
            for obs in self._internal_obstacles:
                self.obs_poly = self.obs_poly.union(obs)
            self.known_space_poly = shapely.geometry.Polygon()
            self.area = 1

        # Initialize the "fast_world"
        segs = []
        segs += [[-500.1, -500.2, -500.3, 500.4]]
        segs += [[-500.3, 500.4, 500.5, 500.6]]
        segs += [[500.5, 500.6, 500.7, -500.8]]
        segs += [[500.7, -500.8, -500.1, -500.2]]

        def add_coords(boundary, segs):
            print(boundary)
            for p0, p1 in zip(boundary.coords, boundary.coords[1:]):
                segs += [[p0[0], p0[1], p1[0], p1[1]]]

        for obs in self.obstacles:
            boundary = obs.boundary
            if isinstance(boundary, geometry.MultiLineString):
                for line in boundary:
                    add_coords(line, segs)
            else:
                add_coords(boundary, segs)

        self.fast_world = gaccel.FastWorld(segs, self.get_all_vertices())

    def compute_iou(self, known_space_poly):
        """Return intersection over union of kown space between
        estimate and ground truth"""
        try:
            intersection = self.known_space_poly.intersection(
                known_space_poly).area
            union = self.known_space_poly.union(known_space_poly).area
            return intersection / union
        except:  # noqa
            print("IoU failed!")
            return 0.0

    @property
    def map_bounds(self):
        """Get the x and y limits of the underlying map.

        Returns:
           xbounds: 2-element tuple [min(x), max(x)]
           ybounds: 2-element tuple [min(y), max(y)]
        """

        if self.boundary is not None:
            xs, ys = self.boundary.exterior.xy
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)
        else:
            # FIXME(gjstein): should not be hard coded
            return (-100, 200), (-100, 200)

        return (xmin, xmax), (ymin, ymax)

    def get_vertex_detection_type_for_vertex(self, robot_pose, vertex):
        """For a given robot pose, return the vertex type from
        the robot's perspective"""
        robot_position = (robot_pose.x, robot_pose.y)
        angle_rad = math.atan2(vertex[1] - robot_pose.y,
                               vertex[0] - robot_pose.x)
        # Add another two lines to the left and right of the corner point and
        # compute intersections.
        s_off = 0.0001
        s_vec = (-math.sin(angle_rad), math.cos(angle_rad))
        l_off = 0.01
        l_vec = (math.cos(angle_rad), math.sin(angle_rad))
        dc_l = (vertex[0] + s_off * s_vec[0] + l_off * l_vec[0],
                vertex[1] + s_off * s_vec[1] + l_off * l_vec[1])
        dc_r = (vertex[0] - s_off * s_vec[0] + l_off * l_vec[0],
                vertex[1] - s_off * s_vec[1] + l_off * l_vec[1])

        point = geometry.Point(vertex)
        vobs = [
            obs for obs in self.strtree.query(point)
            if not isinstance(self._compute_intersection(point, obs),
                              geometry.GeometryCollection)
        ]

        if len(vobs) == 0:
            return 'p'

        def intersect_dist(rp, dc):
            line = geometry.LineString([rp, dc])
            for obs in vobs:
                intr = self._compute_intersection(line, obs)
                if isinstance(intr, geometry.Point):
                    dx = intr.x - rp[0]
                    dy = intr.y - rp[1]
                    return dx * dx + dy * dy
                elif isinstance(intr, geometry.MultiPoint):
                    return min([(p.x - rp[0])**2 + (p.y - rp[1])**2
                                for p in intr])
            return None

        l_int_dist = intersect_dist(robot_position, dc_l)
        r_int_dist = intersect_dist(robot_position, dc_r)
        dx = robot_position[0] - vertex[0]
        dy = robot_position[1] - vertex[1]
        c_dist = dx * dx + dy * dy

        if l_int_dist is not None and r_int_dist is not None:
            if l_int_dist + r_int_dist > 2 * c_dist:
                corner_type = 'c'
            else:
                corner_type = 'i'
            # FIXME(gjstein): should I add a "flat" corner definition?
        elif l_int_dist is None and r_int_dist is None:
            raise ValueError("Point gaps unsupported.")
        elif l_int_dist is None:
            corner_type = 'l'
        elif r_int_dist is None:
            corner_type = 'r'
        else:
            raise ValueError("Point gaps unsupported.")

        return corner_type

    def _compute_intersection(self, line, obs):
        """Computes an intersection between a line and a shapely geometry object.
        The 'intersection' function behaves differently for different classes in
        shapely, and this function is designed to abstract that away."""
        return line.intersection(obs.boundary)

    def get_signed_dist(self, pose):
        """Get the signed distance to obstacles from a point."""
        # Loop through objects, get closest point and return distance
        point = geometry.Point([pose.x, pose.y])
        distance = 1e10
        for obstacle in self._internal_obstacles:
            if obstacle.contains(point):
                obs_dist = -obstacle.exterior.distance(point)
            else:
                obs_dist = obstacle.distance(point)

            distance = min(distance, obs_dist)

        if self.boundary is not None:
            # Ensure that the point is also inside the boundary
            if self.boundary.contains(point):
                boundary_dist = self.boundary.exterior.distance(point)
            else:
                boundary_dist = -self.boundary.distance(point)
            distance = min(distance, boundary_dist)

        return distance

    def get_random_pose(self,
                        xbounds=None,
                        ybounds=None,
                        min_signed_dist=0,
                        max_signed_dist=10000,
                        num_attempts=10000):
        """Get a random pose in the world, respecting the signed distance
        to all the obstacles.

        Each "bound" is a N-element list structured such that:

        > xmin = min(xbounds)
        > xmax = max(xbounds)

        "num_attempts" is the number of trials before an error is raised.
        """

        try:
            xbounds, ybounds = self._map_bounds
        except AttributeError:
            pass

        if xbounds is None or ybounds is None:
            if self.boundary is None:
                raise ValueError("If world.boundary is None, bounds " +
                                 "must be provided.")
            else:
                xbounds, ybounds = self.boundary.exterior.xy

        xmin = min(xbounds)
        xmax = max(xbounds)
        ymin = min(ybounds)
        ymax = max(ybounds)

        counter = 0
        while counter < num_attempts:
            pose = Pose(random.uniform(xmin, xmax), random.uniform(ymin, ymax))
            obs_signed_distance = self.get_signed_dist(pose)
            if obs_signed_distance >= min_signed_dist and obs_signed_distance <= max_signed_dist:
                return pose
            else:
                counter += 1
        else:
            raise ValueError("Could not find random point within bounds")

    def get_all_vertices(self):
        """Loop through all polys and get corners"""

        corner_lists = [
            list(obs.exterior.coords)[:-1] for obs in self.obstacles
        ]
        # Combine the nested lists into a single list of corners
        return sum(corner_lists, [])


class ProposedWorld(WorldBase):
    """Stores the vertices and walls that define a proposed world

    Attributes:
        boundary: boundary of world
        vertices: Noisy Vertex objects
        obstacles: Lines representing walls between vertices
        walls: wall objects between vertices
        covis_memo: dictionary for covisibility between vertices
        obs_memo: dictionary for observations
        neighbor_points: dictionary for neighbor vertices
    """
    def __init__(self,
                 vertices,
                 walls,
                 topology=None,
                 vertex_remapping=None,
                 wall_dict=None):
        self.boundary = None
        self.vertices = vertices
        self.obstacles = [geometry.LineString(wall) for wall in walls]
        self.walls = walls
        self.topology = topology
        self.vertex_remapping = vertex_remapping
        self.wall_dict = wall_dict

        segs = [[w[0][0], w[0][1], w[1][0], w[1][1]] for w in walls]
        segs += [[-500.1, -500.2, -500.3, 500.4]]
        segs += [[-500.3, 500.4, 500.5, 500.6]]
        segs += [[500.5, 500.6, 500.7, -500.8]]
        segs += [[500.7, -500.8, -500.1, -500.2]]

        self.fast_world = gaccel.FastWorld(segs,
                                           [[v[0], v[1]] for v in vertices])

        self.covis_memo = {}
        self.obs_memo = {}
        self.neighbor_points = {}

    def get_vertex_detection_type_for_vertex(self, robot_pose, vertex):
        robot_position = (robot_pose.x, robot_pose.y)

        # Add another two lines to the left and right of the corner point and
        # compute intersections.

        def is_points_near(pA, pB):
            return (abs(pA[1] - pB[1]) < 1e-6 and abs(pA[0] - pB[0]) < 1e-6)

        try:
            ps = self.neighbor_points[vertex]
        except:  # noqa
            ps = np.array([
                obs.coords[1] for obs in self.obstacles
                if is_points_near(obs.coords[0], vertex)
            ] + [
                obs.coords[0] for obs in self.obstacles
                if is_points_near(obs.coords[1], vertex)
            ])

            self.neighbor_points[vertex] = ps

        if len(ps) == 0:
            return 'p'

        rvec = [vertex[0] - robot_position[0], vertex[1] - robot_position[1]]
        vecs = np.array([[p[0] - vertex[0], p[1] - vertex[1]] for p in ps])
        angles = np.arctan2(rvec[0] * vecs[:, 1] - rvec[1] * vecs[:, 0],
                            rvec[0] * vecs[:, 0] + rvec[1] * vecs[:, 1])

        left_walls = [a for a in angles if a > 0]
        if len(left_walls) == 0:
            return 'l'

        right_walls = [a for a in angles if a < 0]
        if len(right_walls) == 0:
            return 'r'

        if max(left_walls) - min(right_walls) < math.pi:
            return 'c'
        else:
            return 'i'

    def _compute_intersection(self, line, obs):
        """Computes an intersection between a line and a shapely geometry object.
        The 'intersection' function behaves differently for different classes in
        shapely, and this function is designed to abstract that away."""
        return line.intersection(obs)

    def get_dist(self, pose):
        """Get the signed distance to obstacles from a point."""
        # Loop through objects, get closest point and return distance
        point = geometry.Point([pose.x, pose.y])
        try:
            return min(
                [obstacle.distance(point) for obstacle in self.obstacles])
        except ValueError:
            return float('inf')

    def get_all_vertices(self):
        """Loop through all polys and get corners"""
        return self.vertices

    def get_inflated_vertices(self, inflation_rad=0.001):
        """Get inflated vertices for computing visibility graph for
        navigation.

        This procedure generates new 'vertices' for every point along."""
        inflated_verts = []
        inflated_obstacles = []

        def is_points_near(pA, pB):
            return (abs(pA[1] - pB[1]) < 1e-8 and abs(pA[0] - pB[0]) < 1e-8)

        def add_vert_and_poly(vert, low_angle, high_angle, num_angles):
            poly_verts = []
            if abs(low_angle - high_angle % (2 * math.pi)) < 0.01:
                is_endpoint = False
            else:
                is_endpoint = True
            for th in np.linspace(low_angle,
                                  high_angle,
                                  num=num_angles,
                                  endpoint=is_endpoint):
                inflated_verts.append((vert[0] + inflation_rad * math.cos(th),
                                       vert[1] + inflation_rad * math.sin(th)))
                poly_verts.append(
                    (vert[0] + 0.95 * inflation_rad * math.cos(th),
                     vert[1] + 0.95 * inflation_rad * math.sin(th)))

            inflated_obstacles.append(geometry.Polygon(poly_verts))

        for vert in self.vertices:
            # Get all enabled walls for this vertex
            walls = [
                wall for wall in self.walls if is_points_near(wall[0], vert)
                or is_points_near(wall[1], vert)
            ]
            if len(walls) == 0:
                add_vert_and_poly(vert, 0.0, 2 * math.pi, num_angles=6)
                continue

            # Compute wall angles_rad
            angles_rad = []
            for wall in walls:
                if is_points_near(wall[0], vert):
                    nvert = wall[1]
                else:
                    nvert = wall[0]
                angles_rad.append(
                    math.atan2(nvert[1] - vert[1], nvert[0] - vert[0]) %
                    (2 * math.pi))

            if len(walls) == 1:
                add_vert_and_poly(vert,
                                  low_angle=angles_rad[0] + 0.01 * math.pi,
                                  high_angle=angles_rad[0] + 1.99 * math.pi,
                                  num_angles=7)
                continue

            angles_rad = sorted(angles_rad)
            poly_verts = []
            for angles in zip(angles_rad, angles_rad[1:] + angles_rad[:1]):
                th = 0.5 * (angles[0] + angles[1])
                if angles[0] > angles[1]:
                    # Handle the 'loop around' condition
                    th += math.pi
                inflated_verts.append((vert[0] + inflation_rad * math.cos(th),
                                       vert[1] + inflation_rad * math.sin(th)))
                poly_verts.append(
                    (vert[0] + 0.95 * inflation_rad * math.cos(th),
                     vert[1] + 0.95 * inflation_rad * math.sin(th)))

            if len(poly_verts) > 3:
                inflated_obstacles.append(geometry.Polygon(poly_verts))
            else:
                inflated_obstacles.append(
                    geometry.LineString(poly_verts).buffer(1e-6))

        return inflated_verts, inflated_obstacles

    def is_covisible(self, vert1, vert2, inflated_obstacles=[]):
        """Returns Bool if two vertices can see one another (respecting the
        underlying geometry and inflated obstacles)."""

        key = (tuple(vert1), tuple(vert2), len(inflated_obstacles))
        if key in self.covis_memo:
            return self.covis_memo[key]

        rkey = (tuple(vert2), tuple(vert1), len(inflated_obstacles))

        pair = np.array([vert1, vert2])
        line = geometry.LineString(pair)

        obs_key = len(self.obstacles) + len(inflated_obstacles)

        if obs_key not in list(self.obs_memo.keys()):
            obs_tree = strtree.STRtree(self.obstacles + inflated_obstacles)
            self.obs_memo[obs_key] = obs_tree
        else:
            obs_tree = self.obs_memo[obs_key]

        for obs in obs_tree.query(line):
            if line.intersects(obs):
                self.covis_memo[key] = False
                self.covis_memo[rkey] = False
                return False

        self.covis_memo[key] = True
        self.covis_memo[rkey] = True
        return True

    def get_visibility_edges_from_verts(self, verts, inflated_obstacles=[]):
        # Loop though all pairs of vertices
        return [
            pair for pair in itertools.combinations(verts, 2)
            if self.is_covisible(pair[0], pair[1], inflated_obstacles)
        ]
