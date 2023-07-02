import random
from common import Pose
import shapely
import shapely.prepared
from shapely import geometry


class World(object):
    """Stores the shapely polygons that define the world (base structure for
    many environments).

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
            self.obs_poly = shapely.geometry.Polygon()
            for obs in self._internal_obstacles:
                self.obs_poly = self.obs_poly.union(obs)
            self.known_space_poly = shapely.geometry.Polygon()
            self.area = 1

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
            # FIXME: should not be hard coded
            return (-100, 200), (-100, 200)

        return (xmin, xmax), (ymin, ymax)

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
