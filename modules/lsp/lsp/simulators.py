"""We define the Simulator class which stores parameters so that certain common
operations, like inflating the occupancy grid, computing simulated sensor
measurements, and plotting are easily computed.
"""
import logging
import math
import time

import common
import environments

from gridmap import laser, mapping, utils
from gridmap.constants import UNOBSERVED_VAL
import lsp.core


class Simulator(object):
    def __init__(self,
                 known_map,
                 goal,
                 args,
                 unity_bridge=None,
                 world=None,
                 verbose=True):
        """args requires a number of values:

        - base_resolution (float) resolution of a single grid cell
        - inflation_radius_m (float) inflation radius of grid (in meters)
        - laser_max_range_m (float) max range of laser scanner (in meters)
        - field_of_view_deg (float) laser scanner field of view
        - laser_scanner_num_points (int) number of points in the sim scan
        - current_seed (int) seed for map generation (only used to name logs)
        """
        # Store some necesasry data and arguments
        self.args = args
        self.goal = goal
        self.resolution = args.base_resolution
        self.inflation_radius = args.inflation_radius_m / self.resolution
        self.frontier_grouping_inflation_radius = 0

        self.laser_max_range_m = args.laser_max_range_m
        self.disable_known_grid_correction = args.disable_known_grid_correction

        # Store the known grid
        self.known_map = known_map.copy()
        self.inflated_known_grid = utils.inflate_grid(
            known_map, inflation_radius=self.inflation_radius)

        # Create the directions object
        self.laser_scanner_num_points = args.laser_scanner_num_points
        self.directions = laser.get_laser_scanner_directions(
            num_points=self.laser_scanner_num_points,
            field_of_view_rad=math.radians(args.field_of_view_deg))

        self.unity_bridge = unity_bridge
        self.world = world
        # self.grid_data_dict = None

        self.verbose = verbose

    def get_laser_scan(self, robot):
        """Get a simulated laser scan."""
        # Get the laser scan
        ranges = laser.simulate_sensor_measurement(
            self.known_map,
            self.directions,
            max_range=self.laser_max_range_m / self.resolution + 2,
            sensor_pose=robot.pose)

        return ranges

    def pose_grid_to_world(self, grid_pose):
        if self.world is None:
            raise ValueError("Cannot convert to world coords if world is None")
        x = grid_pose.x * self.resolution + min(self.world.x)
        y = grid_pose.y * self.resolution + min(self.world.y)
        return common.Pose(x, y, grid_pose.yaw)

    def get_image(self, robot, do_crop=True, do_get_depth=False, do_get_segmentation=False):
        """Get image from unity simulation environment."""
        if self.unity_bridge is None:
            if not do_get_depth and not do_get_segmentation:
                return None
            elif do_get_depth and do_get_segmentation:
                return None, None, None,
            else:
                return None, None

        if self.verbose:
            print("Computing Image")
            t = time.time()

        # Move the vehicle to the pose and get an image
        self.unity_bridge.move_object_to_pose(
            "robot", self.pose_grid_to_world(robot.pose))

        # Get, crop, and orient the image
        pano_image = self.unity_bridge.get_image("robot/pano_camera")
        if do_crop:
            s = pano_image.shape
            pano_image = pano_image[s[0] // 4:3 * s[0] // 4]
        pano_image = environments.utils.convert.image_aligned_to_robot(
            image=pano_image, r_pose=robot.pose)

        if do_get_depth:
            # Get, crop, and orient the image
            pano_depth_image = self.unity_bridge.get_image("robot/pano_depth_camera")
            if do_crop:
                s = pano_depth_image.shape
                pano_depth_image = pano_depth_image[s[0] // 4:3 * s[0] // 4]
            pano_depth_image = environments.utils.convert.image_aligned_to_robot(
                image=pano_depth_image, r_pose=robot.pose)
            pano_depth_image = environments.utils.convert.depths_from_depth_image(
                pano_depth_image)

        if do_get_segmentation:
            pano_segmentation_image = self.unity_bridge.get_image("robot/pano_segmentation_camera")
            if do_crop:
                s = pano_segmentation_image.shape
                pano_segmentation_image = pano_segmentation_image[s[0] // 4:3 * s[0] // 4]
            pano_segmentation_image = environments.utils.convert. \
                image_aligned_to_robot(
                    image=pano_segmentation_image, r_pose=robot.pose)

        if self.verbose:
            print(f"  image time: {time.time() - t}")

        if do_get_depth and do_get_segmentation:
            return pano_image, pano_depth_image, pano_segmentation_image
        elif do_get_depth:
            return pano_image, pano_depth_image
        elif do_get_segmentation:
            return pano_image, pano_segmentation_image
        else:
            return pano_image

    def get_laser_scan_and_update_map(self,
                                      robot,
                                      observed_map,
                                      get_newly_observed=False):
        """Get the simulate laser scan and insert it into the grid."""
        logger = logging.getLogger("simulators")
        stime = time.time()
        ranges = self.get_laser_scan(robot)
        logger.debug(f"time to get laser scan: {time.time() - stime}")

        if not self.disable_known_grid_correction:
            return self._update_map_with_correction(robot, ranges,
                                                    observed_map,
                                                    get_newly_observed)

        # Insert the scan
        stime = time.time()
        observed_map = mapping.insert_scan(observed_map,
                                           self.directions,
                                           laser_ranges=ranges,
                                           max_range=self.laser_max_range_m /
                                           self.resolution,
                                           sensor_pose=robot.pose,
                                           connect_neighbor_distance=2)
        logger.debug(f"time to insert laser scan: {time.time() - stime}")

        # Optionally get and return the visibility mask
        if get_newly_observed:
            newly_observed_grid = mapping.insert_scan(
                0 * observed_map - 1,
                self.directions,
                laser_ranges=ranges,
                max_range=self.laser_max_range_m / self.resolution,
                sensor_pose=robot.pose,
                connect_neighbor_distance=2)

        # Optionally "correct" the grid using the known map. This compensates
        # for errors in the reprojection of the laser scan introduces by the
        # Bresenham line algorithm used for ray tracing.
        if not self.disable_known_grid_correction:
            known = self.known_map.copy()
            mask = (observed_map == UNOBSERVED_VAL)
            known[mask] = UNOBSERVED_VAL
            observed_map = known

            if get_newly_observed:
                known = self.known_map.copy()
                mask = (newly_observed_grid == UNOBSERVED_VAL)
                known[mask] = UNOBSERVED_VAL
                newly_observed_grid = known

        if get_newly_observed:
            return ranges, observed_map, newly_observed_grid
        else:
            return ranges, observed_map

    def _update_map_with_correction(self, robot, ranges, observed_map,
                                    get_newly_observed):
        newly_observed_grid = mapping.insert_scan(
            observed_map,
            self.directions,
            laser_ranges=ranges,
            max_range=self.laser_max_range_m / self.resolution,
            sensor_pose=robot.pose,
            do_only_compute_visibility=True)

        new_visibility_mask = (newly_observed_grid != UNOBSERVED_VAL)
        observed_map[new_visibility_mask] = self.known_map[new_visibility_mask]

        if get_newly_observed:
            newly_observed_grid[new_visibility_mask] = self.known_map[
                new_visibility_mask]
            return ranges, observed_map, newly_observed_grid
        else:
            return ranges, observed_map

    def get_updated_frontier_set(self, inflated_grid, robot, saved_frontiers):
        """Compute the frontiers, store the new ones and compute properties."""
        new_frontiers = lsp.core.get_frontiers(
            inflated_grid,
            group_inflation_radius=self.frontier_grouping_inflation_radius)
        saved_frontiers = lsp.core.update_frontier_set(saved_frontiers,
                                                       new_frontiers)

        lsp.core.update_frontiers_goal_in_frontier(saved_frontiers, self.goal)

        return saved_frontiers

    def get_inflated_grid(self, observed_map, robot):
        """Compute the inflated grid."""
        # Inflate the grid and generate a plan
        inflated_grid = utils.inflate_grid(
            observed_map, inflation_radius=self.inflation_radius)

        # Prevents robot from getting stuck occasionally: sometimes (very
        # rarely) the robot would reveal an obstacle and then find itself
        # within the inflation radius of that obstacle. This should have
        # no side-effects, since the robot is expected to be in free space.
        inflated_grid[int(robot.pose.x), int(robot.pose.y)] = 0

        return inflated_grid
