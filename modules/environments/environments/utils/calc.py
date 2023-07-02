import itertools
import numpy as np
import shapely.geometry
import shapely.ops


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


def _convert_grid_to_poly(grid, resolution, do_full_simplify=True):
    """Takes an occupancy grid and builds a polygon out of all cells that have a value >= 0.5.
    The resolution parameter controls how large each grid cell is in metric space. By
    default, the polygon is 'full simplified', but can be disabled to save computation."""

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

    joined_poly = shapely.ops.cascaded_union(polys)

    if do_full_simplify:
        return full_simplify_shapely_polygon(joined_poly)
    else:
        return joined_poly


def split_semantic_grid_to_polys(occupancy_grid, semantic_grid, semantic_class_index,
                                 resolution, do_compute_walls=True, verbose=False):
    # Compute the shapely polygons for the different classes
    polys = {
        label: _convert_grid_to_poly(semantic_grid == val, resolution)
        for label, val in semantic_class_index.items()
    }
    polys['all_free_space'] = _convert_grid_to_poly(occupancy_grid < 0.5, resolution)

    if verbose:
        for label, poly in polys.items():
            print(f"polys['{label}'].area: {poly.area}")

    if not do_compute_walls:
        return polys

    # Compute the walls
    wall_class_labels = set(semantic_class_index.keys())
    all_walls = polys['all_free_space'].boundary

    # Split the walls by the various semantic classes
    for label in wall_class_labels:
        inf_poly_boundary = polys[label].buffer(0.1 * resolution, 0).boundary
        if isinstance(all_walls, shapely.geometry.GeometryCollection):
            all_walls = shapely.geometry.MultiLineString([ls for ls in all_walls])
        all_walls = shapely.ops.split(
            all_walls, inf_poly_boundary)

    # Split the walls by class
    inflated_polys = {label: poly.buffer(0.2 * resolution, 0)
                      for label, poly in polys.items()}
    walls = {label: [] for label in wall_class_labels}
    walls['base'] = []
    for w in all_walls:
        for label in wall_class_labels:
            if inflated_polys[label].contains(w):
                walls[label].append(w)
                break
        else:
            walls['base'].append(w)

    return polys, walls


def obstacles_and_boundary_from_occupancy_grid(grid, resolution):
    known_space_poly = _convert_grid_to_poly(grid, resolution, do_full_simplify=False)

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
