import math
import numpy as np
import random
import shapely
import shapely.prepared
from shapely import geometry

import vertexnav
from vertexnav.world import World
from unitybridge import UnityBridge


class WorldBuildingUnityBridge(UnityBridge):
    """Connection between World object and unity"""
    def make_world(self, world):
        """Build world in unity"""
        for obstacle in world.obstacles:
            xs, ys = obstacle.exterior.xy

            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                n = max(1, n)
                for i in range(0, len(lst), n):
                    if i == 0:
                        yield lst[i:i + n]
                    else:
                        yield lst[i - 1:i + n]

            for lls in chunks(list(zip(xs, ys)), 10):
                coords = ""
                for x, y in lls:
                    coords += " {} {}".format(y, x)
                message = "main_builder poly" + coords
                print(message)
                self.send_message(message)

        self._add_clutter(world)

    def _add_clutter(self, world):
        """Add clutter to unity sim"""
        for pose in world.clutter_element_poses:
            self.create_object(command_name='clutter',
                               pose=pose,
                               height=random.random() * 6.0 - 4.0)

    def regenerate_clutter(self, world):
        # Clear old clutter
        self.send_message("main_builder reset_clutter")
        # Make new clutter and publish
        world.regenerate_clutter()
        self._add_clutter(world)

    def move_object_to_pose(self, object_name, pose):
        self.send_message("{} move {} {} {} {}".format(object_name, pose.y,
                                                       1.5, pose.x, pose.yaw))


def get_hallway_segment(start_point, end_point, width=20):
    """Gets a hallway segment that extends from a start point to an end
    point (each a 2-element python list or numpy array) with a specified
    width. The return type is a shapely Polygon."""
    sp = np.array(start_point)
    ep = np.array(end_point)
    s_to_e = ep - sp
    s_to_e = s_to_e / np.linalg.norm(s_to_e) * width * 0.5
    s_to_e_perp = np.array([s_to_e[1], -s_to_e[0]])

    return geometry.Polygon([
        sp - s_to_e - s_to_e_perp, sp - s_to_e + s_to_e_perp,
        ep + s_to_e + s_to_e_perp, ep + s_to_e - s_to_e_perp
    ])


def build_hallway_from_path(path, width=20, do_enforce_hallway=True):
    """Creates a simple 'hallway' map by connecting segments that follow a
    provided path. Note that this path does *not* need to be the one that
    the robot ultimately follows. The 'width' parameter controls the width
    of the hallway (and the size of the 'end caps' on each segment).

    The 'do_enforce_hallway' flag ensures that the hallway never intersects
    itself; if set to false, the hallway will be built regardless of the
    properties of the input path."""

    # First collect the individual segments
    segments = []
    for sp, ep in zip(path[:-1], path[1:]):
        segment = get_hallway_segment(sp, ep, width=width)
        segments.append(segment)

    # Loop through segments and ensure that each segment only collides
    # with its immediate proceeding neighbor.
    poly = None
    for s_old, s_new in zip(segments[:-1], segments[1:]):
        if poly is None:
            poly = s_old
        elif do_enforce_hallway and poly.distance(s_new) < 0.0001:
            raise ValueError("Hallway intersects itself.")
        poly = poly.union(s_old)
    # Add the final segment
    poly = poly.union(segments[-1])

    # Handle if the polygon has an interior; any interior 'rings' are
    # converted to Polygons and returned as obstacles. This will only
    # matter if do_enforce_hallway == False.
    obstacles = [
        vertexnav.utils.calc.full_simplify_shapely_polygon(
            geometry.Polygon(interior)) for interior in list(poly.interiors)
    ]

    # Simplify the polygon
    boundary = vertexnav.utils.calc.full_simplify_shapely_polygon(poly)

    return obstacles, boundary


def gen_hallway_path(num_segments, hall_width):
    """Generate a random path (for a hallway)."""
    unit_direction = np.array([1.0, 0.0])
    path = np.array([[0.0, 0.0]])
    for seg_ind in range(num_segments):
        seg_length = random.uniform(2 * hall_width, 5 * hall_width)
        path = np.append(path, [path[-1] + seg_length * unit_direction],
                         axis=0)

        # Randomly rotate the direction
        if random.random() > 0.5:
            unit_direction = np.array([unit_direction[1], unit_direction[0]])
        else:
            unit_direction = np.array([unit_direction[1], -unit_direction[0]])

    return path


class HallwayWorld(World):
    """Constructs a simulated hallway world."""
    def __init__(self,
                 hall_width=20,
                 num_segments=6,
                 num_attempts=10000,
                 num_clutter_elements=0,
                 min_clutter_signed_distance=0.0,
                 max_clutter_signed_distance=1.0,
                 do_enforce_hallway=True):
        # Construct a hallway
        for _ in range(num_attempts):
            path = gen_hallway_path(num_segments, hall_width)

            try:
                obstacles, boundary = build_hallway_from_path(
                    path,
                    width=hall_width,
                    do_enforce_hallway=do_enforce_hallway)
                self.path = path
                super(HallwayWorld, self).__init__(obstacles=obstacles,
                                                   boundary=boundary)
                break
            except ValueError:
                pass

        else:
            raise RuntimeError("Failed to generate hallway map " +
                               "in {} attempts".format(num_attempts))

        # Add clutter (intersects walls)
        self.clutter_element_poses = []
        while len(self.clutter_element_poses) < num_clutter_elements:
            pose = self.get_random_pose(
                min_signed_dist=min_clutter_signed_distance)
            signed_dist = self.get_signed_dist(pose)
            if signed_dist <= max_clutter_signed_distance \
               and signed_dist >= min_clutter_signed_distance:
                self.clutter_element_poses.append(pose)


class OutdoorWorld(World):
    """Constructs a simulated outdoor world with buildings and trees"""
    def __init__(self,
                 hall_width=20,
                 inflate_ratio=0,
                 num_attempts=10000,
                 num_clutter_elements=30,
                 min_signed_distance=25.0,
                 max_signed_distance=40.0,
                 min_clutter_signed_distance=15.0,
                 max_clutter_signed_distance=40.0,
                 num_buildings=8):

        self.num_clutter_elements = num_clutter_elements
        self.min_signed_distance = min_signed_distance
        self.max_signed_distance = max_signed_distance
        self.min_clutter_signed_distance = min_clutter_signed_distance
        self.max_clutter_signed_distance = max_clutter_signed_distance
        self.clutter_element_data = None

        obs_command = []
        obstacles = []

        def is_new_obs_acceptable(all_obstacles, new_obs):
            if len(all_obstacles) == 0:
                return True, random.random() + min_signed_distance

            dists = [obs.distance(new_obs) for obs in all_obstacles]
            current_closest = min(dists)
            tot_dist = sum(sorted(dists)[0:3])

            if (current_closest > min_signed_distance
                    and current_closest < max_signed_distance):
                return True, 0 * current_closest + tot_dist / len(dists[0:3])
            else:
                return False, None

        def get_random_building():
            bld_name = random.choice(["A1", "E1", "E5"])
            # rot_deg = random.choice([0, 90, 180, 270])
            rot_deg = random.random() * 360.0
            pot_obs_cmd = [
                bld_name, 300 * random.random() - 150,
                300 * random.random() - 20, 2.0 + 1.5 * random.random(),
                2.0 + 1.5 * random.random(), rot_deg
            ]
            if bld_name == "A1":
                pot_obs_cmd[3] *= 0.5
                pot_obs_cmd[4] *= 0.5
                pot_obs = _get_bld_A1(*pot_obs_cmd[1:])
            elif bld_name == "E1":
                pot_obs = _get_bld_E1(*pot_obs_cmd[1:])
            elif bld_name == "E5":
                pot_obs_cmd[3] *= 1.5
                pot_obs_cmd[4] *= 1.5
                pot_obs = _get_bld_E5(*pot_obs_cmd[1:])
            else:
                raise ValueError(
                    "Building type '{}' not allowed.".format(bld_name))

            return pot_obs_cmd, pot_obs

        while len(obs_command) < num_buildings:
            blds = [get_random_building() for _ in range(1000)]
            dists = [is_new_obs_acceptable(obstacles, b[1]) for b in blds]
            accepted = [(b[0], b[1], d[1]) for b, d in zip(blds, dists)
                        if d[0]]

            if len(accepted) == 0:
                continue

            pot_obs_cmd, pot_obs, _ = min(accepted, key=lambda a: a[2])

            print((pot_obs, obs_command))
            if is_new_obs_acceptable(obstacles, pot_obs):
                obs_command.append(pot_obs_cmd)
                obstacles.append(pot_obs)

        minx = 1e8
        maxx = -1e8
        miny = 1e8
        maxy = -1e8
        for obs in obstacles:
            x1, y1, x2, y2 = obs.bounds
            minx = min(minx, x1, x2)
            maxx = max(maxx, x1, x2)
            miny = min(miny, y1, y2)
            maxy = max(maxy, y1, y2)

        self._map_bounds = ((minx, maxx), (miny, maxy))

        self.obs_command = obs_command
        super(OutdoorWorld, self).__init__(obstacles=obstacles)

        # Overwrite default values for area
        self.boundary_poly = shapely.geometry.Polygon([(minx, miny),
                                                       (maxx, miny),
                                                       (maxx, maxy),
                                                       (minx, maxy)])
        self.known_space_poly = self.boundary_poly
        for obs in obstacles:
            self.known_space_poly = self.known_space_poly.difference(obs)
        self.area = self.known_space_poly.area

        # Generate the clutter
        self.regenerate_clutter()

    def get_nearby_clutter(self, robot_pose, dist_threshold):
        def pose_dist(pa, pb):
            return math.sqrt((pa.x - pb.x)**2 + (pa.y - pb.y)**2)

        return [
            p for p in self.clutter_element_poses
            if pose_dist(p, robot_pose) <= dist_threshold
        ]

    def regenerate_clutter(self):
        # Add clutter
        self.clutter_element_poses = []
        while len(self.clutter_element_poses) < self.num_clutter_elements:
            pose = self.get_random_pose(
                min_signed_dist=self.min_clutter_signed_distance,
                max_signed_dist=self.max_clutter_signed_distance)
            signed_dist = self.get_signed_dist(pose)
            if signed_dist <= self.max_clutter_signed_distance \
               and signed_dist >= self.min_clutter_signed_distance:
                self.clutter_element_poses.append(pose)

    @property
    def map_bounds(self):
        return self._map_bounds

    def compute_iou(self, known_space_poly):
        """For the outdoor world, the IoU is computed differently."""
        try:
            known_space_poly = known_space_poly.buffer(0.1,
                                                       resolution=0,
                                                       cap_style=3,
                                                       join_style=2)
            interior_ksp = shapely.geometry.Polygon()
            if isinstance(known_space_poly, shapely.geometry.Polygon):
                for poly in list(known_space_poly.interiors):
                    interior_ksp = interior_ksp.union(geometry.Polygon(poly))

            interior_ksp = interior_ksp.buffer(0.1,
                                               resolution=0,
                                               cap_style=3,
                                               join_style=2)

            intersection = self.obs_poly.intersection(interior_ksp).area
            union = self.obs_poly.union(interior_ksp).area
            return intersection / union
        except:  # noqa
            print("IoU failed!")
            return 0.00001


class OutdoorBuildingUnityBridge(WorldBuildingUnityBridge):
    """Unity Bridge for outdoor world"""
    def make_world(self, world, scale=10.0):
        for cmd in world.obs_command:
            message = 'main_builder bld {} {} {} {} {} {}'.format(*cmd)
            self.send_message(message)

        self._add_clutter(world)

    def _add_clutter(self, world):
        for pose in world.clutter_element_poses:
            self.create_object(command_name=random.choice(['tree',
                                                           'tree_alt']),
                               pose=pose,
                               height=0.0)

    def move_object_to_pose(self, object_name, pose):
        self.send_message("{} move {} {} {} {}".format(object_name, pose.y,
                                                       2.5, pose.x, pose.yaw))


class OccupancyGridWorld(World):
    """Use occupancy grid to improve planning efficiency"""
    def __init__(self,
                 grid,
                 map_data,
                 num_breadcrumb_elements=500,
                 min_breadcrumb_signed_distance=4.0):
        self.grid = (1.0 - grid.T)  # Differences in occupancy value
        self.map_data = map_data
        self.resolution = map_data['resolution']
        print(f"OGW resolution: {map_data['resolution']}")

        obstacles, boundary = vertexnav.utils.calc.obstacles_and_boundary_from_occupancy_grid(
            self.grid, self.resolution)

        self.x = (np.arange(0, self.grid.shape[0]) + 0.0) * self.resolution
        self.y = (np.arange(0, self.grid.shape[1]) + 0.0) * self.resolution

        super(OccupancyGridWorld, self).__init__(obstacles=obstacles,
                                                 boundary=boundary)

        # Add clutter (intersects walls)
        self.breadcrumb_element_poses = []
        while len(self.breadcrumb_element_poses) < num_breadcrumb_elements:
            pose = self.get_random_pose(
                min_signed_dist=min_breadcrumb_signed_distance,
                semantic_label='goal_path')
            signed_dist = self.get_signed_dist(pose)
            if signed_dist >= min_breadcrumb_signed_distance:
                self.breadcrumb_element_poses.append(pose)

    def get_random_pose(self,
                        xbounds=None,
                        ybounds=None,
                        min_signed_dist=0,
                        num_attempts=10000,
                        semantic_label=None):
        """Get a random pose in the world, respecting the signed distance
        to all the obstacles.

        Each "bound" is a N-element list structured such that:

        > xmin = min(xbounds)
        > xmax = max(xbounds)

        "num_attempts" is the number of trials before an error is raised.

        """

        for _ in range(num_attempts):
            pose = super(OccupancyGridWorld,
                         self).get_random_pose(xbounds, ybounds,
                                               min_signed_dist, num_attempts)
            if semantic_label is None:
                return pose

            # FIXME: Unsure quite why this flip is necessary...
            pose_cell_x = np.argmin(np.abs(self.y - pose.x))
            pose_cell_y = np.argmin(np.abs(self.x - pose.y))

            grid_label_ind = self.map_data['semantic_grid'][pose_cell_x,
                                                            pose_cell_y]
            if grid_label_ind == self.map_data['semantic_labels'][
                    semantic_label]:
                return pose
        else:
            raise ValueError("Could not find random point within bounds")

    def get_grid_from_poly(self, known_space_poly, proposed_world=None):
        known_space_poly = known_space_poly.buffer(self.resolution / 2)
        mask = -1 * np.ones(self.grid.shape)

        for ii in range(mask.shape[0]):
            for jj in range(mask.shape[1]):
                p = shapely.geometry.Point(self.y[jj], self.x[ii])
                if known_space_poly.contains(p):
                    mask[ii, jj] = 1

        out_grid = -1.0 * np.ones(mask.shape)
        # out_grid[mask == 1] = self.grid[mask == 1]
        # out_grid[np.logical_and(mask == 1, self.grid == 1)] = 0.0
        # out_grid[np.logical_and(mask == 1, self.grid == 0)] = 1.0
        out_grid[mask == 1] = 0.0

        if proposed_world is not None:
            for v in proposed_world.vertices:
                cell_y = np.argmin(np.abs(self.y - v[0]))
                cell_x = np.argmin(np.abs(self.x - v[1]))
                out_grid[cell_x, cell_y] = 1.0

            for w in proposed_world.walls:
                ys = np.linspace(w[0][0], w[1][0], 100)
                xs = np.linspace(w[0][1], w[1][1], 100)

                for x, y in zip(xs, ys):
                    cell_x = np.argmin(np.abs(self.x - x))
                    cell_y = np.argmin(np.abs(self.y - y))
                    out_grid[cell_x, cell_y] = 1.0

        return out_grid.T


def get_poly_from_base_points(func):
    def transform_points(tx, tz, sx, sy, rot_deg):
        points = func()
        st = math.sin(math.radians(-rot_deg))
        ct = math.cos(math.radians(-rot_deg))

        scaled_points = [(sy * p[0], sx * p[1]) for p in points]

        rotated_points = [(ct * p[0] + st * p[1], -st * p[0] + ct * p[1])
                          for p in scaled_points]

        return geometry.Polygon([(tz + rp[0], tx + rp[1])
                                 for rp in rotated_points])

    return transform_points


@get_poly_from_base_points
def _get_bld_E1():
    return [
        (-10.5, 10.5),
        (3.5, 10.5),
        (3.5, 3.5),
        (10.5, 3.5),
        (10.5, -10.5),
        (-10.5, -10.5),
    ]


@get_poly_from_base_points
def _get_bld_E5():
    return [
        (7.0, -3.5),
        (7.0, 3.5),
        (-7.0, 3.5),
        (-7.0, -3.5),
    ]


@get_poly_from_base_points
def _get_bld_A1():
    return [
        (-14.1, 14.1),
        (14.1, 14.1),
        (14.1, -14.1),
        (-14.1, -14.1),
    ]
    return [
        (-14.3, 14.3),
        (14.3, 14.3),
        (14.3, -14.3),
        (-14.3, -14.3),
    ]


def get_world_guardian_center():
    """Guardian Center for UNITY"""
    # A square
    # bld1 = geometry.Polygon([])

    bld = []
    # Regions numbered in "reading order": across rows starting at top

    # Region 1
    bld += [
        geometry.Polygon([
            (63.44, -52.08),
            (75.94, -64.54),
            (73.1, -67.32),
            (79.35, -73.49),
            (73.79, -79.23),
            (73.39, -79.01),
            (66.16, -86.11),
            (59.17, -79.22),
            (66.34, -72.06),
            (54.97, -60.49),
        ])
    ]
    bld += [_get_bld_E1(tx=-48.9, tz=84.1, sx=0.5952, sy=1.1291, rot_deg=45)]

    # Region 2
    bld += [_get_bld_E1(tx=-9.3, tz=105.3, sx=1.1618, sy=0.6061, rot_deg=45)]
    bld += [_get_bld_A1(tx=21.4, tz=123.1, sx=0.337, sy=1.19, rot_deg=45)]
    bld += [
        geometry.Polygon([(147.55, 21.09), (164.28, 4.46), (146.53, -13.4),
                          (145.84, -12.71), (129.28, -29.27), (119.48, -19.47),
                          (136.06, -2.89), (129.78, 3.39)])
    ]

    # Region 3
    bld += [_get_bld_E1(tx=7.6, tz=-1.8, sx=0.5524, sy=1.2164, rot_deg=90.0)]
    bld += [_get_bld_E5(tx=-27.43, tz=38.77, sx=1.714, sy=1.718, rot_deg=45)]
    bld += [_get_bld_E1(tx=-37.1, tz=15.3, sx=1.2067, sy=0.5881, rot_deg=45)]
    bld += [_get_bld_E1(tx=-5.9, tz=48, sx=1.1453, sy=0.6232, rot_deg=45)]
    bld += [_get_bld_E5(tx=30.7, tz=19.9, sx=3.2857, sy=0.8638, rot_deg=90)]

    # Region 4
    bld += [
        geometry.Polygon([
            (73.23, 11.27),
            (55.23, 29.27),
            (63.03, 37.07),
            (80.23, 19.87),
            (92.35, 31.99),
            (94.73, 29.61),
            (102.14, 37.02),
            (106.8, 32.36),
            (84.59, 10.15),
            (78.39, 16.35),
        ])
    ]

    # Region 5
    bld += [_get_bld_E1(tx=63.22, tz=5.13, sx=1.0717, sy=0.6243, rot_deg=90)]
    bld += [_get_bld_E1(tx=72.15, tz=25.87, sx=0.5714, sy=1.239, rot_deg=90)]
    bld += [_get_bld_E1(tx=99.92, tz=27.49, sx=0.6333, sy=1.1067, rot_deg=90)]
    bld += [_get_bld_E1(tx=108.9, tz=5.64, sx=1.16, sy=0.5971, rot_deg=90)]

    # Region 6
    bld += [_get_bld_E5(tx=66.3, tz=75.6, sx=1.644, sy=1.660, rot_deg=-90)]
    bld += [_get_bld_A1(tx=98.3, tz=70.1, sx=0.8596, sy=1.2158, rot_deg=90)]
    bld += [_get_bld_E1(tx=75.4, tz=99, sx=0.619, sy=1.147, rot_deg=90)]
    bld += [_get_bld_E1(tx=99.7, tz=99, sx=0.619, sy=1.147, rot_deg=90)]

    # Region 7
    bld += [_get_bld_E5(tx=85, tz=128.5, sx=1.573, sy=1.7157, rot_deg=90)]
    bld += [_get_bld_A1(tx=108, tz=145.7, sx=0.8596, sy=0.4924, rot_deg=90)]

    # Region 8
    bld += [_get_bld_A1(tx=141, tz=19, sx=0.8772, sy=0.4737, rot_deg=90)]
    bld += [_get_bld_A1(tx=163, tz=29, sx=0.4211, sy=0.9480, rot_deg=90)]
    bld += [_get_bld_A1(tx=184, tz=18, sx=1.2193, sy=0.4386, rot_deg=90)]

    # Region 9
    bld += [_get_bld_E5(tx=149, tz=59, sx=1.145, sy=2.5, rot_deg=90)]
    bld += [_get_bld_A1(tx=182, tz=56, sx=0.5614, sy=0.7368, rot_deg=90)]
    bld += [_get_bld_A1(tx=137, tz=85, sx=1.0351, sy=0.3864, rot_deg=90)]
    bld += [_get_bld_E1(tx=184, tz=91, sx=1.1667, sy=0.5952, rot_deg=90)]

    # Region 10
    bld += [_get_bld_E1(tx=137, tz=134, sx=1.1669, sy=0.5971, rot_deg=90)]
    bld += [_get_bld_A1(tx=161, tz=131, sx=1.2357, sy=0.35087, rot_deg=0)]
    bld += [_get_bld_E1(tx=185, tz=134, sx=1.1669, sy=0.5971, rot_deg=90)]

    return World(obstacles=bld)


def get_world_mit_north_court():
    """MIT north court environment"""
    bld = []

    bld.append(
        geometry.Polygon([
            (-57.25, 82.95),
            (-67.5, 170.7),
            (-97.2, 100.3),
            (-80.8, 96.7),
            (-87, 81.2),
            (-92.5, 82.9),
            (-144.5, -38.3),
            (-128.6, -44.7),
            (-105, 12.2),
            (-101.7, 11.2),
            (-75.3, 76),
            (-83.8, 79.6),
            (-78.2, 91.6),
        ]))

    return World(obstacles=bld)
