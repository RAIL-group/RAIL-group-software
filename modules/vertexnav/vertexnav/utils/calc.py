import itertools
import math
import numpy as np
from skimage import measure
import shapely
import shapely.ops

import vertexnav
"""Helper functions for doing calculations used in many files"""

# Define some helper indices
yind = 0
heightind = 1
xind = 2
yawind = 3


def data_to_world_pose(d):
    x = d[xind]
    y = d[yind]
    yaw = d[yawind] * math.pi / 180.0
    pose = vertexnav.Pose(x, y, yaw)
    return pose


def eucl_dist(pos1, pos2):
    return math.sqrt(eucl_dist_sq(pos1, pos2))


def eucl_dist_sq(pos1, pos2):
    return (pos1[1] - pos2[1])**2 + (pos1[0] - pos2[0])**2


def directions_vec(num_im_cols):
    """Returns an array of 'direction vectors' for a panoramic image
    from Unity"""

    angles_rad = np.linspace(-math.pi, math.pi, num_im_cols + 1)[:-1]
    directions = np.vstack((np.cos(angles_rad), np.sin(angles_rad)))
    return (directions, angles_rad)


def range_bearing_vecs(max_range, num_range, num_bearing):
    vec_range = np.linspace(start=0.0, stop=max_range, num=num_range + 1)
    vec_range = vec_range[1:]
    _, vec_bearing = directions_vec(num_bearing)

    return vec_range, vec_bearing


def detect_peaks(image, is_circular_boundary=False, peak_thresh=None):

    if is_circular_boundary:
        ncols = image.shape[1]
        image = np.concatenate((image, image), axis=1)
        assert image.shape[1] == ncols * 2

    labels = measure.label(image >= (peak_thresh - 0.1))

    peaks = []
    for ii in range(1, labels.max().max() + 1):
        tmp = np.zeros(image.shape)
        mask = (labels == ii)
        tmp[mask] = image[mask]
        ind = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
        if tmp[ind] > peak_thresh:
            peaks.append(ind)

    if is_circular_boundary:
        return np.array([(coord[0], coord[1] % ncols) for coord in peaks
                         if coord[1] >= ncols / 2 and coord[1] < 3 * ncols / 2
                         ])

    return peaks


# State estimation and distance measures
def m_distance(pos_1, pos_2, inv_noise):
    dpos = np.array([[pos_1[0] - pos_2[0]], [pos_1[1] - pos_2[1]]])
    M_dist = math.sqrt(np.matmul(dpos.T, np.matmul(inv_noise, dpos)))
    return M_dist


def transform_local_to_world(det_theta, det_r, cov_rt, r_pose):
    world_theta = det_theta + r_pose.yaw
    cov_r = cov_rt[0, 0]
    r_sq_cov_theta = (det_r**2) * cov_rt[1, 1]
    sin_th = math.sin(world_theta)
    cos_th = math.cos(world_theta)
    sin_th_sq = sin_th**2
    cos_th_sq = cos_th**2
    Q11 = r_sq_cov_theta * sin_th_sq + cov_r * cos_th_sq
    Q22 = r_sq_cov_theta * cos_th_sq + cov_r * sin_th_sq
    Q12 = (cov_r - r_sq_cov_theta) * cos_th * sin_th
    position = [r_pose.x + det_r * cos_th, r_pose.y + det_r * sin_th]
    return position, np.array([[Q11, Q12], [Q12, Q22]])


def smooth_path(path, segment_steps=50, smooth_factor=1.0 / 3):
    steps = segment_steps
    points = [
        (ii * 1.0 * seg[1][0] / steps + (1 - ii * 1.0 / steps) * seg[0][0],
         ii * 1.0 * seg[1][1] / steps + (1 - ii * 1.0 / steps) * seg[0][1])
        for seg in zip(path[:-1], path[1:]) for ii in range(steps)
    ]
    points = np.array(points)

    # Smooth the points
    sw = int(steps * smooth_factor)
    if sw > 0:
        avg_kernel = np.ones(sw) / sw
        points[:, 0] = np.convolve(
            np.pad(points[:, 0], (sw // 2, sw - sw // 2 - 1), 'edge'),
            avg_kernel, 'valid')
        points[:, 1] = np.convolve(
            np.pad(points[:, 1], (sw // 2, sw - sw // 2 - 1), 'edge'),
            avg_kernel, 'valid')

    return points


def full_simplify_shapely_polygon(poly):
    """This function simplifies a polygon, removing any colinear points.
    Though Shapely has this functionality built-in, it won't remove the
    "start point" of the polygon, even if it's colinear."""

    if isinstance(poly, shapely.geometry.MultiPolygon) or isinstance(
            poly, shapely.geometry.GeometryCollection):
        return shapely.geometry.MultiPolygon(
            [full_simplify_shapely_polygon(p) for p in poly])

    poly = poly.simplify(0.001, preserve_topology=True)
    # The final point is removed, since shapely will auto-close polygon
    points = np.array(poly.exterior.coords)
    if (points[-1] == points[0]).all():
        points = points[:-1]

    def is_colinear(p1, p2, p3, tol=1e-6):
        """Checks if the area formed by a triangle made of the three points
        is less than a tolerance value."""
        return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] *
                   (p1[1] - p2[1])) < tol

    if is_colinear(points[0], points[1], points[-1]):
        poly = shapely.geometry.Polygon(points[1:])

    return poly


def obstacles_and_boundary_from_occupancy_grid(grid, resolution):
    print("Computing obstacles")
    print(resolution)

    known_space_poly = shapely.geometry.Polygon()

    polys = []
    for index, val in np.ndenumerate(grid):
        if val < 0.5:
            continue

        y, x = index
        y *= resolution
        x *= resolution
        y -= 0.5 * resolution
        x -= 0.5 * resolution
        r = resolution
        polys.append(
            shapely.geometry.Polygon([(x, y), (x + r, y), (x + r, y + r),
                                      (x, y + r)
                                      ]).buffer(0.001 * resolution, 0))

    known_space_poly = shapely.ops.cascaded_union(polys)

    def get_obstacles(poly):
        if isinstance(poly, shapely.geometry.MultiPolygon):
            return list(
                itertools.chain.from_iterable([get_obstacles(p)
                                               for p in poly]))

        obstacles = [
            full_simplify_shapely_polygon(shapely.geometry.Polygon(interior))
            for interior in list(poly.interiors)
        ]

        # Simplify the polygon
        boundary = full_simplify_shapely_polygon(poly)

        obstacles.append(boundary)

        return obstacles

    obs = get_obstacles(known_space_poly)
    obs.sort(key=lambda x: x.area, reverse=True)
    return obs[1:], obs[0]

    # # Old versions
    # return get_obstacles(known_space_poly), None
    # return [], get_obstacles(known_space_poly)[0]
