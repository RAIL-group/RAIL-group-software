# import matplotlib
# matplotlib.use('Qt5Agg')  # noqa: E702
import matplotlib
from matplotlib.patches import Ellipse
import math
import numpy as np
import shapely
import vertexnav

LINEWIDTH = 1.5
SCATTERSIZE = 4.0
"""
A bunch of functions to facilitate plotting results
"""


def plot_proposed_world(ax,
                        world,
                        do_show_points=False,
                        do_plot_visibility=False,
                        robot_pose=None):
    for obstacle in world.obstacles:
        x, y = obstacle.xy
        ax.plot(x, y, 'k', linewidth=LINEWIDTH)

    if do_show_points and len(world.vertices):
        points = np.array(world.vertices)
        if points.size > 0:
            ax.plot(points[:, 0], points[:, 1], 'k.', markersize=SCATTERSIZE)

    if do_plot_visibility:
        # FIXME(gjstein): don't hard-code inflation radius
        inf_verts = world.get_inflated_vertices(inflation_rad=4)
        if robot_pose is not None:
            inf_verts.append((robot_pose.x, robot_pose.y))
        vis_edges = world.get_visibility_edges_from_verts(inf_verts)
        for line in vis_edges:
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                    'b',
                    alpha=0.2)


def plot_world(ax, world, do_show_points=False, alpha=1.0):
    for obstacle in world.obstacles:
        x, y = obstacle.exterior.xy
        if do_show_points:
            ax.plot(x, y, 'ko-', linewidth=LINEWIDTH, alpha=alpha)
        else:
            ax.plot(x, y, 'k', linewidth=LINEWIDTH, alpha=alpha)

    if isinstance(world, vertexnav.environments.dungeon.DungeonWorld):
        counter = 0
        for p in world.clutter_element_poses:
            if counter % 2 == 1:
                # Make a box
                th = np.linspace(0.25 * math.pi, 2.25 * math.pi, 5)
                points = np.array([3 * np.cos(th) + p.x, 3 * np.sin(th) + p.y])
            else:
                # Make a pipe
                th = np.linspace(0.0, 2 * math.pi, 33)
                points = np.array([1 * np.cos(th) + p.x, 1 * np.sin(th) + p.y])
            ax.plot(points[0],
                    points[1],
                    color='yellow',
                    linewidth=LINEWIDTH,
                    alpha=min(alpha + 0.2, 0.8))
            counter += 1
    elif isinstance(world, vertexnav.environments.simulated.OutdoorWorld):
        counter = 0
        for p in world.clutter_element_poses:
            # Make a tree
            th = np.linspace(0.0, 2 * math.pi, 33)
            points = np.array([1 * np.cos(th) + p.x, 1 * np.sin(th) + p.y])
            ax.plot(points[0],
                    points[1],
                    color='orange',
                    linewidth=LINEWIDTH,
                    alpha=alpha + 0.2)
            counter += 1


def plot_noisy_wall(ax, wall):
    lvp = wall.left_vertex.position
    rvp = wall.right_vertex.position
    prob = wall.prob_exists
    ax.plot([lvp[0], rvp[0]], [lvp[1], rvp[1]],
            color=[1 - prob, prob, 0],
            alpha=0.5 * prob + 0.1)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def plot_corners_position_overhead(ax, obs, plt_format='mo', do_plt_cov=True):
    for gap in obs:
        ax.plot(gap.position[0], gap.position[1], plt_format, mfc='none')
        if do_plt_cov and gap.cov is not None:
            nstd = 1
            cov = gap.cov
            vals, vecs = eigsorted(cov)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=(gap.position[0], gap.position[1]),
                          width=w,
                          height=h,
                          angle=theta,
                          color='black')
            ell.set_facecolor('b')
            ell.set_edgecolor(None)
            ell.set_alpha(0.2)
            ax.add_artist(ell)


def plot_shapely_linestring(ax, ls, color=[0.25, 0.25, 1.0], alpha=1.0):
    if isinstance(ls, shapely.geometry.MultiLineString):
        [plot_shapely_linestring(ax, line, color, alpha) for line in ls]
        return

    x, y = ls.xy
    ax.plot(x, y, color=color, alpha=alpha, linewidth=0.2)


def plot_polygon(ax, poly, color=[0.0, 0.0, 1.0], alpha=1.0):
    # Plotting code from: https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html

    if isinstance(poly, shapely.geometry.MultiPolygon):
        [plot_polygon(ax, p, color, alpha) for p in poly]
        return

    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(ob.coords)
        codes = np.ones(n, dtype=matplotlib.path.Path.code_type) \
            * matplotlib.path.Path.LINETO
        codes[0] = matplotlib.path.Path.MOVETO
        return codes

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by Shapely's
        # analytic methods have the proper coordinate order, no need to sort.
        vertices = np.concatenate([np.asarray(polygon.exterior)] +
                                  [np.asarray(r) for r in polygon.interiors])
        codes = np.concatenate([ring_coding(polygon.exterior)] +
                               [ring_coding(r) for r in polygon.interiors])
        return matplotlib.path.Path(vertices, codes)

    try:
        path = pathify(poly)
        patch = matplotlib.patches.PathPatch(path,
                                             facecolor=color,
                                             linewidth=0,
                                             alpha=alpha)
        ax.add_patch(patch)
    except:  # noqa
        pass


def plot_linestring(ax, linestring, **kwargs):

    if isinstance(linestring, shapely.geometry.GeometryCollection):
        return
    if isinstance(linestring, shapely.geometry.MultiLineString):
        [plot_linestring(ax, ls, **kwargs) for ls in linestring]
        return

    x, y = linestring.xy
    ax.plot(x, y, **kwargs)
