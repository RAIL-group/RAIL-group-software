import argparse
import environments.simulated
import vertexnav
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import shapely.geometry


class Dungeon():
    """Funtionality for building and returning shapely polygon of environment.

    Arguments:
        rooms (Room obj): list of rooms to build environment from.
        scale (int): width of hallways
        inflate_ratio (float): how much to inflate obstacles

    Attributes:
        scale (int): width of hallways
        inflate_ratio (float): how much to inflate obstacles
        grid (np.array): occupancy grid of environment (1 is free)
        xc (int): x bound
        yc (int): y bound
    """
    def __init__(self, rooms, scale=10, inflate_ratio=0):
        self.scale = scale
        self.inflate_ratio = inflate_ratio
        if self.inflate_ratio >= 1.0 / 2.0:
            raise ValueError("Inflation ratio too large.")

        if _do_rooms_overlap(rooms):
            raise ValueError

        xs = [x for room in rooms for x in room.xbounds]
        xbounds = (min(xs) - 1, max(xs) + 1)
        ys = [y for room in rooms for y in room.ybounds]
        ybounds = (min(ys) - 1, max(ys) + 1)

        self.xc = np.arange(xbounds[0], xbounds[1] + 1)
        self.yc = np.arange(ybounds[0], ybounds[1] + 1)

        self.grid = np.zeros((self.xc.size, self.yc.size))
        self.rooms = []
        for room in rooms:
            self._add_room(room)

        config = random.choice([0, 1, 2, 3])

        if config == 0:
            # Cycle between three; final room connected at random to another
            self._add_room_connection(self.rooms[0], self.rooms[1])
            self._add_room_connection(self.rooms[1], self.rooms[2])
            self._add_room_connection(self.rooms[2], self.rooms[0])
            self._add_room_connection(self.rooms[3],
                                      random.choice(self.rooms[0:3]))
        elif config == 1:
            # Cycle between four
            self._add_room_connection(self.rooms[0], self.rooms[1])
            self._add_room_connection(self.rooms[1], self.rooms[2])
            self._add_room_connection(self.rooms[2], self.rooms[3])
            self._add_room_connection(self.rooms[3], self.rooms[0])
        elif config == 2:
            # No cycle
            self._add_room_connection(self.rooms[0], self.rooms[1])
            self._add_room_connection(self.rooms[1], self.rooms[2])
            self._add_room_connection(self.rooms[2], self.rooms[3])
        elif config == 3:
            # Two cycles connected by another line
            self._add_room_connection(self.rooms[0], self.rooms[1])
            self._add_room_connection(self.rooms[0], self.rooms[1])
            self._add_room_connection(self.rooms[1], self.rooms[2])
            self._add_room_connection(self.rooms[2], self.rooms[3])
            self._add_room_connection(self.rooms[2], self.rooms[3])
        else:
            raise ValueError("Configuration not possible")

        # Now check for runs of a certain length
        from itertools import groupby

        def len_iter(items):
            return sum(1 for _ in items)

        def consecutive_one(data):
            v = [len_iter(run) for val, run in groupby(data) if val == 1]
            if len(v):
                return max(v)
            else:
                return 0

        for row in self.grid:
            if consecutive_one(row) > 11:
                raise ValueError
        for col in self.grid.T:
            if consecutive_one(col) > 11:
                raise ValueError

        # Prune impossible corners
        if np.logical_and(
                np.logical_and(self.grid[1:, :-1] == 0, self.grid[:-1,
                                                                  1:] == 0),
                np.logical_and(self.grid[1:, 1:] == 1,
                               self.grid[:-1, :-1] == 1)).any():
            raise ValueError
        if np.logical_and(
                np.logical_and(self.grid[1:, :-1] == 1, self.grid[:-1,
                                                                  1:] == 1),
                np.logical_and(self.grid[1:, 1:] == 0,
                               self.grid[:-1, :-1] == 0)).any():
            raise ValueError

    def compute_obstacles_and_boundary(self):
        """Return the shapely polygons for the boundary and all obstacles"""
        print("Computing obstacles")
        print((self.inflate_ratio))

        known_space_poly = shapely.geometry.Polygon()

        cx = 0.5 * (self.xc[0] + self.xc[-1])
        cy = 0.5 * (self.yc[0] + self.yc[-1])

        if self.inflate_ratio > 0:
            s = self.scale * 1.0 / (1 + self.inflate_ratio)
            print(("Scale: {}".format(s)))
        else:
            s = 1

        for index, val in np.ndenumerate(self.grid):
            if val < 1:
                continue

            y, x = index
            y *= s
            x *= s
            y -= (0.5 + cy) * s
            x -= (0.5 + cx) * s
            poly = shapely.geometry.Polygon([(x, y), (x + s, y),
                                             (x + s, y + s), (x, y + s)])
            known_space_poly = known_space_poly.union(poly)

        if self.inflate_ratio > 0:
            known_space_poly = known_space_poly.buffer(s * self.inflate_ratio /
                                                       2,
                                                       0,
                                                       cap_style=3,
                                                       join_style=2)

        # Handle if the polygon has an interior; any interior 'rings' are
        # converted to Polygons and returned as obstacles. This will only
        # matter if do_enforce_hallway == False.
        obstacles = [
            vertexnav.utils.calc.full_simplify_shapely_polygon(
                shapely.geometry.Polygon(interior))
            for interior in list(known_space_poly.interiors)
        ]

        # Simplify the polygon
        boundary = vertexnav.utils.calc.full_simplify_shapely_polygon(
            known_space_poly)

        return obstacles, boundary

    def _compute_poly_and_walls(self):
        """Compute known space polygon and walls for the environment"""

        known_space_poly = shapely.geometry.Polygon()

        for index, val in np.ndenumerate(self.grid):
            if val < 1:
                continue

            s = self.scale
            y, x = index
            y *= s
            x *= s
            y -= 0.5 * s
            x -= 0.5 * s
            poly = shapely.geometry.Polygon([(x, y), (x + s, y),
                                             (x + s, y + s), (x, y + s)])
            known_space_poly = known_space_poly.union(poly)

        walls = []
        for ls in known_space_poly.boundary:
            print(ls)
            for pa, pb in zip(ls.coords[1:], ls.coords[:-1]):
                walls.append((pa, pb))

        known_space_exterior = vertexnav.utils.calc.full_simplify_shapely_polygon(
            known_space_poly)
        interiors = [
            vertexnav.utils.calc.full_simplify_shapely_polygon(
                shapely.geometry.Polygon(interior))
            for interior in list(known_space_poly.interiors)
        ]
        known_space_poly = known_space_exterior
        for interior in interiors:
            known_space_poly = known_space_poly.difference(interior)

        return known_space_poly, walls

    def _add_room(self, room):
        """Adds a room and updates the grid"""

        is_room_x = np.logical_and(self.xc >= min(room.xbounds),
                                   self.xc <= max(room.xbounds))
        is_room_y = np.logical_and(self.yc >= min(room.ybounds),
                                   self.yc <= max(room.ybounds))
        self.grid[np.outer(is_room_x, is_room_y)] = 1
        self.rooms.append(room)

    def _add_room_connection(self, room_a, room_b):
        """Connects rooms with hallways and updates the grid"""
        if random.random() > 0.5:
            start, end = room_a.get_random_point(), room_b.get_random_point()
        else:
            start, end = room_b.get_random_point(), room_a.get_random_point()

        start = [
            np.argwhere(start[0] == self.xc)[0],
            np.argwhere(start[1] == self.yc)[0]
        ]
        end = [
            np.argwhere(end[0] == self.xc)[0],
            np.argwhere(end[1] == self.yc)[0]
        ]

        if abs(start[0] - end[0]) > 6 or abs(start[1] - end[1]) > 6:
            raise ValueError

        # Draw first a vertical then a horizontal line
        point = start
        while not point[0] == end[0]:
            point[0] = point[0] - int(math.copysign(1, point[0] - end[0]))
            self.grid[point[0], point[1]] = 1
        while not point[1] == end[1]:
            point[1] = point[1] - int(math.copysign(1, point[1] - end[1]))
            self.grid[point[0], point[1]] = 1


class Room():
    """Funtionality for defining polygon of rooms that make up environment.

    Arguments:
        xbounds (tuple of ints): x bounds of a room
        ybounds (tuple of ints): y bounds of a room

    Attributes:
        xbounds (tuple of ints): x bounds of a room
        ybounds (tuple of ints): y bounds of a room
        poly (shapely polygon): polygon defining room
    """
    def __init__(self, xbounds, ybounds):
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.poly = shapely.geometry.Polygon([(min(xbounds), min(ybounds)),
                                              (max(xbounds), min(ybounds)),
                                              (max(xbounds), max(ybounds)),
                                              (min(xbounds), max(ybounds))])

    def get_random_point(self):
        "Returns uniformly sampled position from room" ""
        return [
            random.randint(min(self.xbounds) + 1,
                           max(self.xbounds) - 1),
            random.randint(min(self.ybounds) + 1,
                           max(self.ybounds) - 1)
        ]


def _do_rooms_overlap(rooms):
    """Return if rooms overlap in map generation"""
    for r_pair in itertools.combinations(rooms, 2):
        if r_pair[0].poly.buffer(2.05, 1).intersects(r_pair[1].poly):
            return True

    return False


def _get_dungeon(scale=10, inflate_ratio=0):
    """Returns sample map"""
    while True:
        rooms = []

        for _ in range(4):
            xc = random.randint(0, 14)
            width = random.randint(2, 3)
            yc = random.randint(0, 14)
            height = random.randint(2, 3)
            rooms.append(Room([xc, xc + width], [yc, yc + height]))
        try:
            return Dungeon(rooms, scale=scale, inflate_ratio=inflate_ratio)
        except Exception as e:
            if len(str(e)) > 2:
                raise
            pass


class DungeonWorld(vertexnav.world.World):
    """Implementation of world class to build Dungeon World"""
    def __init__(self,
                 hall_width=20,
                 inflate_ratio=0.3,
                 num_attempts=10000,
                 num_clutter_elements=40,
                 min_clutter_signed_distance=0.5,
                 max_clutter_signed_distance=1.5,
                 min_interlight_distance=30,
                 min_light_to_wall_distance=9,
                 random_seed=None):

        self.breadcrumb_type = None

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Construct a hallway
        for _ in range(num_attempts):
            dungeon_obj = _get_dungeon(scale=hall_width,
                                       inflate_ratio=inflate_ratio)

            try:
                obstacles, boundary = dungeon_obj.compute_obstacles_and_boundary(
                )
                super(DungeonWorld, self).__init__(obstacles=obstacles,
                                                   boundary=boundary)
                break
            except Exception as e:
                if len(str(e)) > 2:
                    print(e)
                    raise
                pass
        else:
            raise RuntimeError("Failed to generate hallway map " +
                               f"in {num_attempts} attempts")

        # Add clutter (intersects walls)
        self.clutter_element_poses = []
        while len(self.clutter_element_poses) < num_clutter_elements:
            pose = self.get_random_pose(
                min_signed_dist=min_clutter_signed_distance)
            signed_dist = self.get_signed_dist(pose)
            if signed_dist <= max_clutter_signed_distance \
               and signed_dist >= min_clutter_signed_distance:
                self.clutter_element_poses.append(pose)

        counter = 0
        self.clutter_element_data = []  # XYZWHD
        for p in self.clutter_element_poses:
            if counter % 2:
                self.clutter_element_data.append(
                    ['box', p.x, p.y, 0.0, 2.25, 2.25, 2.25])
            else:
                self.clutter_element_data.append(
                    ['cylinder', p.x, p.y, 0.0, 1.0, 1.0, 2.25])
            counter += 1

        self.breadcrumb_element_poses = []

        self.light_poses = environments.simulated._generate_light_poses(
            world=self,
            min_interlight_distance=min_interlight_distance,
            min_light_to_wall_distance=min_light_to_wall_distance)

        self.ceiling_poses = environments.simulated._generate_ceiling_poses(
            self.boundary)


def show_test_map():
    """Plots sample map"""
    ax = plt.gca()
    grid = _get_dungeon(inflate_ratio=0.25)
    img = ax.imshow(grid.grid)
    img.set_cmap('binary')
    poly, walls = grid._compute_poly_and_walls()

    vertexnav.plotting.plot_polygon(plt.gca(), poly, alpha=0.2)

    for w in walls:
        plt.plot((w[0][0], w[1][0]), (w[0][1], w[1][1]))
    plt.show()


def parse_args():
    """Define the command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a random map.')
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    show_test_map()
