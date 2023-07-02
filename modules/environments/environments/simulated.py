import math
import numpy as np
import random
import shapely
import shapely.prepared
import tempfile

from .utils.calc import obstacles_and_boundary_from_occupancy_grid
from .world import World
from unitybridge import UnityBridge


def eucl_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class WorldBuildingUnityBridge(UnityBridge):
    """Connection between World object and unity"""
    def make_world(self, world, scale=10.0):
        self.do_buffer = True

        self.send_message("main_builder floor")

        if hasattr(world, 'map_data') and 'walls' in world.map_data.keys():
            for wall_class in world.map_data['walls'].keys():
                for wall in world.map_data['walls'][wall_class]:
                    self._make_wall(wall, wall_class)
        else:
            self._make_obstacles(world, scale)

        if hasattr(world, 'clutter_element_poses'):
            for pose in world.clutter_element_poses:
                self.create_object(command_name='clutter',
                                   pose=pose,
                                   height=random.random() * 0.5)

        if hasattr(world, 'table_poses_sizes'):
            table_pose_sizes = np.array(world.table_poses_sizes) * world.resolution
            for x, y, size_x, size_y in table_pose_sizes:
                self.send_message(f'main_builder table {x} {y} {size_x} {size_y}', pause=0.001)

        if hasattr(world, 'breadcrumb_element_poses'):
            if not hasattr(world, 'breadcrumb_type'):
                world.breadcrumb_type = None
            if world.breadcrumb_type is None:
                breadcrumb_command = 'breadcrumb'
            elif isinstance(world.breadcrumb_type, str):
                breadcrumb_command = f'breadcrumb_{world.breadcrumb_type}'
            else:
                raise ValueError('world.breadcrumb_type expected to be str or None. '
                                 f'Value is: {world.breadcrumb_type}')
            for pose in world.breadcrumb_element_poses:
                self.create_object(command_name=breadcrumb_command,
                                   pose=pose,
                                   height=random.random() * 0.001)

        if hasattr(world, 'light_poses'):
            for x, y in world.light_poses:
                self.send_message(f'main_builder light {x * self.sim_scale} {y * self.sim_scale}', pause=0.001)

        if hasattr(world, 'ceiling_poses'):
            for x, y, s in world.ceiling_poses:
                self.send_message(f'main_builder ceiling {x * self.sim_scale} {y * self.sim_scale} '
                                  f'{s * self.sim_scale}', pause=0.001)

        self.do_buffer = False
        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            for message in self.messages:
                temp_file.write(message + "\n")

        self.send_message(f"main_builder file {temp_file.name}", 0.0)
        self.unity_listener.parse_string()

    def move_object_to_pose(self, object_name, pose, pause=-1):
        if pause <= 0:
            self.send_message("{} move_respond {} {} {} {}".format(
                object_name, pose.x * self.sim_scale, pose.y * self.sim_scale,
                1.5, pose.yaw), pause=pause)
            self.unity_listener.parse_string()
        else:
            self.send_message("{} move {} {} {} {}".format(
                object_name, pose.x * self.sim_scale, pose.y * self.sim_scale,
                1.5, pose.yaw), pause=pause)

    def _make_obstacles(self, world, scale):
        for obstacle in world.obstacles:
            points = obstacle.exterior.coords
            spoints = points[1:] + points[-1:]
            for pa, pb in zip(points, spoints):
                coords = ""
                nsegs = int(eucl_dist(pa, pb) / scale + 0.5)
                nsegs = max(nsegs, 1)
                xs = np.linspace(pa[0], pb[0], nsegs + 1, endpoint=True)
                ys = np.linspace(pa[1], pb[1], nsegs + 1, endpoint=True)

                for x, y in zip(xs, ys):
                    coords += f" {x * self.sim_scale} {y * self.sim_scale}"

                message = "main_builder dungeon_poly" + coords
                self.send_message(message)

    def _make_wall(self, wall, wall_class):
        points = wall.coords
        spoints = points[1:] + points[-1:]
        for pa, pb in zip(points, spoints):
            coords = ""
            nsegs = int(eucl_dist(pa, pb) / 10 + 0.5)
            nsegs = max(nsegs, 1)
            xs = np.linspace(pa[0], pb[0], nsegs + 1, endpoint=True)
            ys = np.linspace(pa[1], pb[1], nsegs + 1, endpoint=True)

            for y, x in zip(xs, ys):
                coords += f" {x * self.sim_scale} {y * self.sim_scale}"

            message = f"main_builder poly_{wall_class}" + coords
            self.send_message(message)


class OccupancyGridWorld(World):
    """Use occupancy grid to improve planning efficiency"""
    def __init__(self,
                 grid,
                 map_data,
                 num_breadcrumb_elements=500,
                 min_breadcrumb_signed_distance=4.0,
                 breadcrumb_type=None,
                 min_interlight_distance=3,
                 min_light_to_wall_distance=2,
                 max_attempts_lights=10000):
        self.grid = (1.0 - grid.T)  # Differences in occupancy value
        self.map_data = map_data
        self.resolution = map_data['resolution']

        obstacles, boundary = obstacles_and_boundary_from_occupancy_grid(
            self.grid, self.resolution)

        self.x = (np.arange(0, self.grid.shape[0]) + 0.0) * self.resolution
        self.y = (np.arange(0, self.grid.shape[1]) + 0.0) * self.resolution

        super(OccupancyGridWorld, self).__init__(obstacles=obstacles,
                                                 boundary=boundary)

        self.breadcrumb_type = breadcrumb_type
        self.breadcrumb_element_poses = []
        while len(self.breadcrumb_element_poses) < num_breadcrumb_elements:
            pose = self.get_random_pose(
                min_signed_dist=min_breadcrumb_signed_distance,
                semantic_label='goal_path')
            signed_dist = self.get_signed_dist(pose)
            if signed_dist >= min_breadcrumb_signed_distance:
                self.breadcrumb_element_poses.append(pose)

        # If there are tables, light poses aren't generated above them since tables are obstacles.
        # To fix this, we need to create a world without tables and generate light poses in this world.
        # Sometimes, this also affects ceiling poses, and hence ceiling poses are generated accordingly.
        if 'tables' in self.map_data.keys():
            self.table_poses_sizes = self.map_data['tables']
            semantic_grid = self.map_data['semantic_grid']
            grid_no_tables = (semantic_grid < self.map_data['semantic_labels']['clutter']).astype(float)
            grid_no_tables = (1.0 - grid_no_tables.T)
            obstacles, boundary = obstacles_and_boundary_from_occupancy_grid(
                grid_no_tables, self.resolution)
            world = World(obstacles, boundary)
            self.light_poses = _generate_light_poses(world=world,
                                                     min_interlight_distance=min_interlight_distance,
                                                     min_light_to_wall_distance=min_light_to_wall_distance,
                                                     max_attempts=max_attempts_lights)
            self.ceiling_poses = _generate_ceiling_poses(world.known_space_poly)
        else:
            self.light_poses = _generate_light_poses(world=self,
                                                     min_interlight_distance=min_interlight_distance,
                                                     min_light_to_wall_distance=min_light_to_wall_distance,
                                                     max_attempts=max_attempts_lights)
            self.ceiling_poses = _generate_ceiling_poses(self.known_space_poly)

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


def _generate_ceiling_poses(known_space_poly, tile_size=2.5):
    x, y = known_space_poly.exterior.xy
    x_min, y_min = min(x), min(y)
    x_max, y_max = max(x), max(y)
    ceiling_poses = []
    for xx in np.arange(x_min, x_max + tile_size, tile_size):
        for yy in np.arange(y_min, y_max + tile_size, tile_size):
            tile_poly = shapely.geometry.Polygon([[xx - tile_size / 2, yy - tile_size / 2],
                                                  [xx + tile_size / 2, yy - tile_size / 2],
                                                  [xx + tile_size / 2, yy + tile_size / 2],
                                                  [xx - tile_size / 2, yy + tile_size / 2]])
            if known_space_poly.intersects(tile_poly):
                ceiling_poses.append([xx, yy, tile_size])

    return ceiling_poses


def _generate_light_poses(world, min_interlight_distance, min_light_to_wall_distance,
                          max_attempts=10000):
    new_pose = world.get_random_pose(min_signed_dist=min_light_to_wall_distance)
    light_poses = [[new_pose.x, new_pose.y]]
    for _ in range(max_attempts):
        new_pose = world.get_random_pose(min_signed_dist=min_light_to_wall_distance)
        interlight_distance_squared = (np.subtract(light_poses, [new_pose.x, new_pose.y])**2).sum(1)
        if np.all(interlight_distance_squared >= min_interlight_distance**2):
            light_poses.append([new_pose.x, new_pose.y])

    return light_poses
