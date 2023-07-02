"""
This function stores everything pertaining to learned subgoal planning and frontiers.
"""

from collections import namedtuple
import logging
import math
import numpy as np
import scipy.ndimage
import skimage.measure
import itertools
import time

from gridmap.constants import (COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL,
                               OBSTACLE_THRESHOLD)

import lsp
from gridmap import planning
import gridmap.utils
import lsp_accel

IS_FROM_LAST_CHOSEN_REWARD = 0 * 10.0


class Frontier(object):
    def __init__(self, points):
        """Initialized with a 2xN numpy array of points (the grid cell
        coordinates of all points on frontier boundary)."""
        inds = np.lexsort((points[0, :], points[1, :]))
        sorted_points = points[:, inds]
        self.props_set = False
        self.is_from_last_chosen = False
        self.is_obstructed = False
        self.prob_feasible = 1.0
        self.delta_success_cost = 0.0
        self.exploration_cost = 0.0
        self.negative_weighting = 0.0
        self.positive_weighting = 0.0

        self.counter = 0
        self.last_observed_pose = None

        # Any duplicate points should be eliminated (would interfere with
        # equality checking).
        dupes = []
        for ii in range(1, sorted_points.shape[1]):
            if (sorted_points[:, ii - 1] == sorted_points[:, ii]).all():
                dupes += [ii]
        self.points = np.delete(sorted_points, dupes, axis=1)

        # Compute and cache the hash
        self.hash = hash(self.points.tobytes())

    def set_props(self,
                  prob_feasible,
                  is_obstructed=False,
                  delta_success_cost=0,
                  exploration_cost=0,
                  positive_weighting=0,
                  negative_weighting=0,
                  counter=0,
                  last_observed_pose=None,
                  did_set=True):
        self.props_set = did_set
        self.just_set = did_set
        self.prob_feasible = prob_feasible
        self.is_obstructed = is_obstructed
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.positive_weighting = positive_weighting
        self.negative_weighting = negative_weighting
        self.counter = counter
        self.last_observed_pose = last_observed_pose

    @property
    def centroid(self):
        return self.get_centroid()

    def get_centroid(self):
        """Returns the point that is the centroid of the frontier"""
        centroid = np.mean(self.points, axis=1)
        return centroid

    def get_frontier_point(self):
        """Returns the point that is on the frontier that is closest to the
        actual centroid"""
        center_point = np.mean(self.points, axis=1)
        norm = np.linalg.norm(self.points - center_point[:, None], axis=0)
        ind = np.argmin(norm)
        return self.points[:, ind]

    def get_distance_to_point(self, point):
        norm = np.linalg.norm(self.points - point[:, None], axis=0)
        return norm.min()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)


def get_frontiers(occupancy_grid, group_inflation_radius=0):
    """Get froniers from the map.

    Frontiers exist at the boundary between free and unknown space. The
    points that make up the frontiers exist in the *unknown* portion of
    the space. This helps avoid some planning issues later on; if frontiers
    are generated in the *known* portion of the map, they may obstruct
    the robot's path and erroneously rule out regions of the map.

    We compute the frontiers using connected components. Masked by all the
    frontiers, a map should confine an agent to the observed region.
    """
    filtered_grid = scipy.ndimage.maximum_filter(np.logical_and(
        occupancy_grid < OBSTACLE_THRESHOLD, occupancy_grid == FREE_VAL), size=3)
    frontier_point_mask = np.logical_and(filtered_grid,
                                         occupancy_grid == UNOBSERVED_VAL)

    if group_inflation_radius < 1:
        inflated_frontier_mask = frontier_point_mask
    else:
        inflated_frontier_mask = gridmap.utils.inflate_grid(
            frontier_point_mask,
            inflation_radius=group_inflation_radius,
            obstacle_threshold=0.5,
            collision_val=1.0) > 0.5

    # Group the frontier points into connected components
    labels, nb = scipy.ndimage.label(inflated_frontier_mask)

    # Extract the frontiers
    frontiers = set()
    for ii in range(nb):
        raw_frontier_indices = np.where(
            np.logical_and(labels == (ii + 1), frontier_point_mask))
        frontiers.add(
            Frontier(
                np.concatenate((raw_frontier_indices[0][None, :],
                                raw_frontier_indices[1][None, :]),
                               axis=0)))

    return frontiers


def mask_grid_with_frontiers(occupancy_grid, frontiers, do_not_mask=None):
    """Mask grid cells in the provided occupancy_grid with the frontier points
    contained with the set of 'frontiers'. If 'do_not_mask' is provided, and
    set to either a single frontier or a set of frontiers, those frontiers are
    not masked."""

    if do_not_mask is not None:
        # Ensure that 'do_not_mask' is a set
        if isinstance(do_not_mask, Frontier):
            do_not_mask = set([do_not_mask])
        elif not isinstance(do_not_mask, set):
            raise TypeError("do_not_mask must be either a set or a Frontier")
        masking_frontiers = frontiers - do_not_mask
    else:
        masking_frontiers = frontiers

    masked_grid = occupancy_grid.copy()
    for frontier in masking_frontiers:
        masked_grid[frontier.points[0, :],
                    frontier.points[1, :]] = COLLISION_VAL

    return masked_grid


def _eucl_dist(p1, p2):
    """Helper to compute Euclidean distance."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _get_nearest_feasible_frontier(frontier, reference_frontier_set):
    """Returns the nearest 'feasible' frontier from a reference set."""
    f_gen = [(of, _eucl_dist(of.get_centroid(), frontier.get_centroid()))
             for of in reference_frontier_set if of.prob_feasible > 0.0]
    if len(f_gen) == 0:
        return None, 1e10
    else:
        return min(f_gen, key=lambda fd: fd[1])


def update_frontier_set(old_set, new_set, max_dist=None, chosen_frontier=None):
    """Updates an old set of frontiers with a new set of frontiers.

    If a frontier persists, it is kept. If a new frontier appears, it is added.
    Everything is done with python set operations. Finally, if a
    "chosen_frontier" is passed, any frontier that derives its properties (i.e.
    is closest to) that frontier from the old set has its 'is_from_last_chosen'
    property set to true.
    """

    # Update the 'just_set' and 'is_from_last_chosen' properties
    for frontier in old_set:
        frontier.just_set = False
        frontier.is_from_last_chosen = False

    # Shallow copy of the set
    old_set = old_set.copy()

    # These are the frontiers that will not appear in the new set
    outgoing_frontier_set = old_set - new_set
    added_frontier_set = new_set - old_set
    if max_dist is not None:
        # Loop through the newly added_frontier_set and set properties based
        # upon the outgoing_frontier_set
        for af in added_frontier_set:
            nearest_frontier, nearest_frontier_dist = (
                _get_nearest_feasible_frontier(
                    frontier=af,
                    reference_frontier_set=outgoing_frontier_set,
                ))
            if nearest_frontier_dist < max_dist:
                af.set_props(
                    prob_feasible=nearest_frontier.prob_feasible,
                    delta_success_cost=nearest_frontier.delta_success_cost,
                    exploration_cost=nearest_frontier.exploration_cost,
                    did_set=False)
                try:
                    af.image = nearest_frontier.image
                    af.mask = nearest_frontier.mask
                    af.goal_loc_x_mat = nearest_frontier.goal_loc_x_mat
                    af.goal_loc_y_mat = nearest_frontier.goal_loc_y_mat
                except AttributeError:
                    pass

                if nearest_frontier == chosen_frontier:
                    af.is_from_last_chosen = True
            else:
                af.set_props(prob_feasible=1.0,
                             delta_success_cost=0.0,
                             exploration_cost=0.0,
                             did_set=False)

    # Remove frontier_set that don't appear in the new set
    old_set.difference_update(outgoing_frontier_set)

    # Add the new frontier_set
    old_set.update(added_frontier_set)

    return old_set


def update_frontiers_properties_known(inflated_known_grid,
                                      inflated_observed_grid,
                                      all_frontiers,
                                      new_frontiers,
                                      start_pose,
                                      end_pose,
                                      downsample_factor=1):
    if not new_frontiers:
        return

    # If the goal is in known space, all frontiers are successful
    # and the success cost is identically zero.
    goal_visible = goal_in_range(inflated_observed_grid, start_pose, end_pose,
                                 all_frontiers)
    if goal_visible:
        for f in new_frontiers:
            f.set_props(prob_feasible=1.0,
                        delta_success_cost=0.0,
                        positive_weighting=1.0)
        return

    # Determine if any success frontiers
    inflated_mixed_grid = np.ones_like(inflated_known_grid)
    inflated_mixed_grid[np.logical_and(
        inflated_known_grid == FREE_VAL,
        inflated_observed_grid == UNOBSERVED_VAL)] = UNOBSERVED_VAL

    known_goal_distances = get_goal_distances(inflated_mixed_grid, end_pose,
                                              new_frontiers, downsample_factor)
    success_frontiers = [
        f for (f, kgd) in known_goal_distances.items() if kgd < 1e8
    ]
    failure_frontiers = [
        f for (f, kgd) in known_goal_distances.items() if kgd >= 1e8
    ]

    # Compute and set the success costs
    if success_frontiers:
        observed_goal_distances = get_goal_distances(inflated_observed_grid,
                                                     end_pose,
                                                     success_frontiers,
                                                     downsample_factor)
        for f, ogd in observed_goal_distances.items():
            if ogd > 1e8:
                f.set_props(prob_feasible=0.0)
            else:
                kgd = known_goal_distances[f]
                f.set_props(prob_feasible=1.0, delta_success_cost=kgd - ogd)

    if failure_frontiers:
        # Get the cost grid from the robot in the inflated_known_grid
        if downsample_factor > 1:
            inflated_known_grid = skimage.measure.block_reduce(
                inflated_known_grid, (downsample_factor, downsample_factor),
                np.min)
            inflated_mixed_grid = skimage.measure.block_reduce(
                inflated_mixed_grid, (downsample_factor, downsample_factor),
                np.min)
        known_cost_grid = planning.compute_cost_grid_from_position(
            inflated_known_grid,
            start=[
                start_pose.x // downsample_factor,
                start_pose.y // downsample_factor
            ],
            only_return_cost_grid=True)

        unk_regions = (inflated_mixed_grid == UNOBSERVED_VAL)
        labels, nb = scipy.ndimage.label(unk_regions)

        # Loop through frontiers and for each
        for f in failure_frontiers:
            # Figure out which label that frontier matches
            fp = f.points // downsample_factor
            flabel = labels[fp[0, :], fp[1, :]].max()
            cost_region = known_cost_grid[labels == flabel]
            min_cost = cost_region.min()
            max_cost = cost_region.max()
            if min_cost > 1e8:
                f.set_props(prob_feasible=0.0)
                f.is_obstructed = True
            else:
                if max_cost > 1e8:
                    cost_region[cost_region > 1e8] = 0
                    max_cost = cost_region.max()

                exploration_cost = 2 * downsample_factor * (max_cost -
                                                            min_cost)
                f.set_props(prob_feasible=0.0,
                            exploration_cost=exploration_cost)


def update_frontiers_weights_known(inflated_known_grid,
                                   inflated_observed_grid,
                                   all_frontiers,
                                   new_frontiers,
                                   start_pose,
                                   end_pose,
                                   downsample_factor=1):
    for frontier in new_frontiers:
        if frontier.prob_feasible > 0.5:
            # Compute the positive weighting
            frontier.positive_weighting = lsp.utils.calc.compute_frontier_positive_weighting(
                inflated_observed_grid, inflated_known_grid, all_frontiers,
                frontier, start_pose, end_pose)
        else:
            # Compute the negative weighting
            frontier.negative_weighting = lsp.utils.calc.compute_frontier_negative_weighting(
                inflated_observed_grid, inflated_known_grid, all_frontiers, frontier,
                start_pose, end_pose, frontier.exploration_cost)


def goal_in_range(grid, robot_pose, goal_pose, frontiers):
    goal_visible = any(
        f.get_distance_to_point(np.array([int(goal_pose.x),
                                          int(goal_pose.y)])) < 1.5
        for f in frontiers)
    goal_visible = (goal_visible or
                    grid[int(goal_pose.x), int(goal_pose.y)] != UNOBSERVED_VAL)
    return goal_visible


def update_frontiers_goal_in_frontier(all_frontiers, end_pose):
    """This function checks to see if the goal point is inside a frontier. If
    it is, all frontiers are set as leading to the goal (and
    props_set <- True)."""

    is_goal_in_frontier = False
    for f in all_frontiers:
        if [int(end_pose.x), int(end_pose.y)] in f.points.T.tolist():
            is_goal_in_frontier = True

    if is_goal_in_frontier:
        for f in all_frontiers:
            if f.props_set is False:
                f.set_props(prob_feasible=1.0)


FrontWithPoint = namedtuple('FrontWithPoint', ['frontier', 'point'])


def get_frontier_distances(grid, frontiers, downsample_factor=1):
    """get_frontier_distances takes an occupancy_grid and returns a dictionary. The
    number of elemnts in it is the number of non repeating combinations of
    frontiers so that every frontier pair is a key to the dictionary. Each key
    is a frozenset of two frontiers. The value in the dictionary is a touple
    that contatains two elements. The first element is the distance between the
    two frontiers that make up the key, and the second element is the shortest
    path between the frontiers stored as a numpy array.
    """
    if len(frontiers) <= 1:
        return None

    occupancy_grid = np.copy(grid)
    frontier_distances = dict()

    frontier_with_point_list = [
        FrontWithPoint(frontier=f,
                       point=f.get_frontier_point() // downsample_factor)
        for f in frontiers
    ]

    all_frontier_points = np.concatenate([f.points for f in frontiers], axis=1)

    occupancy_grid[occupancy_grid == UNOBSERVED_VAL] = COLLISION_VAL
    occupancy_grid[all_frontier_points[0, :],
                   all_frontier_points[1, :]] = FREE_VAL

    if downsample_factor > 1:
        occupancy_grid = skimage.measure.block_reduce(
            occupancy_grid, (downsample_factor, downsample_factor), np.min)

    # I only need the upper triangular block of the pairs. This means that I
    # don't need the final fwp_1 (since it would only be comparing against
    # itself) and I use the enumerate function to select only a subset of the
    # fwp_2 entries.
    for ind, fwp_1 in enumerate(frontier_with_point_list[:-1]):
        # Compute the cost grid for the first frontier
        start = fwp_1.frontier.points // downsample_factor
        cost_grid = planning.compute_cost_grid_from_position(
            occupancy_grid,
            start=start,
            use_soft_cost=False,
            only_return_cost_grid=True)
        for fwp_2 in frontier_with_point_list[ind + 1:]:
            ff_set = frozenset([fwp_1.frontier, fwp_2.frontier])
            fpoints = fwp_2.frontier.points // downsample_factor
            cost = downsample_factor * (cost_grid[fpoints[0, :],
                                                  fpoints[1, :]].min())
            frontier_distances[ff_set] = cost

    return frontier_distances


def get_robot_distances(grid, robot_pose, frontiers, downsample_factor=1):
    """take in occupancy grid and robot position and returns a dictionary relating a frontier to it's distance
    from the robot, and the path corresponding to that distance"""
    occupancy_grid = np.copy(grid)
    robot_distances = dict()
    if len(frontiers) <= 0:
        return [0, np.array([[]])]

    # Properly mask the occupancy grid
    for frontier in frontiers:
        occupancy_grid[frontier.points[0, :], frontier.points[1, :]] = FREE_VAL
    occupancy_grid[occupancy_grid == UNOBSERVED_VAL] = COLLISION_VAL
    if downsample_factor > 1:
        occupancy_grid = skimage.measure.block_reduce(
            occupancy_grid, (downsample_factor, downsample_factor), np.min)

    cost_grid = planning.compute_cost_grid_from_position(
        occupancy_grid,
        start=[
            robot_pose.x // downsample_factor,
            robot_pose.y // downsample_factor
        ],
        use_soft_cost=False,
        only_return_cost_grid=True)

    # Compute the cost for each frontier
    for frontier in frontiers:
        f_pt = frontier.get_frontier_point() // downsample_factor
        cost = cost_grid[f_pt[0], f_pt[1]]

        if math.isinf(cost):
            cost = 100000000000
            frontier.set_props(prob_feasible=0.0, is_obstructed=True)
            frontier.just_set = False

        robot_distances[frontier] = downsample_factor * cost

    return robot_distances


def get_goal_distances(grid, goal_pose, frontiers, downsample_factor=1):
    """take in occupancy grid and goal position and returns a dictionary relating a frontier to it's distance
    from the goal, and the path corresponding to that distance"""
    goal_distances = dict()
    if len(frontiers) <= 0:
        return [0, np.array([[]])]

    occupancy_grid = np.copy(grid)
    occupancy_grid[occupancy_grid == FREE_VAL] = COLLISION_VAL
    occupancy_grid[occupancy_grid == UNOBSERVED_VAL] = FREE_VAL

    if downsample_factor > 1:
        occupancy_grid = skimage.measure.block_reduce(
            occupancy_grid, (downsample_factor, downsample_factor), np.min)

    # Compute the cost grid
    cost_grid = planning.compute_cost_grid_from_position(
        occupancy_grid,
        start=[
            goal_pose.x // downsample_factor, goal_pose.y // downsample_factor
        ],
        use_soft_cost=False,
        only_return_cost_grid=True)

    # Compute the cost for each frontier
    for frontier in frontiers:
        fpts = frontier.points // downsample_factor
        cost = downsample_factor * (cost_grid[fpts[0, :], fpts[1, :]].min())

        if math.isinf(cost):
            cost = 100000000000
            frontier.set_props(prob_feasible=0.0, is_obstructed=True)
            frontier.just_set = False

        goal_distances[frontier] = cost

    return goal_distances


class FState(object):
    """Used to conviently store the 'state' during recursive cost search.
    """
    def __init__(self, new_frontier, distances, old_state=None):
        nf = new_frontier
        p = nf.prob_feasible
        # Success cost
        try:
            sc = nf.delta_success_cost + distances['goal'][nf]
        except KeyError:
            sc = nf.delta_success_cost + distances['goal'][nf.id]
        # Exploration cost
        ec = nf.exploration_cost

        if old_state is not None:
            self.frontier_list = old_state.frontier_list + [nf]
            # Store the old frontier
            of = old_state.frontier_list[-1]
            # Known cost (travel between frontiers)
            try:
                kc = distances['frontier'][frozenset([nf, of])]
            except KeyError:
                kc = distances['frontier'][frozenset([nf.id, of.id])]
            self.cost = old_state.cost + old_state.prob * (kc + p * sc +
                                                           (1 - p) * ec)
            self.prob = old_state.prob * (1 - p)
        else:
            # This is the first frontier, so the robot must accumulate a cost of getting to the frontier
            self.frontier_list = [nf]
            # Known cost (travel to frontier)
            try:
                kc = distances['robot'][nf]
            except KeyError:
                kc = distances['robot'][nf.id]

            if nf.is_from_last_chosen:
                kc -= IS_FROM_LAST_CHOSEN_REWARD
            self.cost = kc + p * sc + (1 - p) * ec
            self.prob = (1 - p)

    def __lt__(self, other):
        return self.cost < other.cost


def get_ordering_cost(subgoals, distances):
    """A helper function to compute the expected cost of a particular ordering.
    The function takes an ordered list of subgoals (the order in which the robot
    aims to explore beyond them). Consistent with the subgoal planning API,
    'distances' is a dictionary with three keys: 'robot' (a dict of the
    robot-subgoal distances), 'goal' (a dict of the goal-subgoal distances), and
    'frontier' (a dict of the frontier-frontier distances)."""
    fstate = None
    for s in subgoals:
        fstate = FState(s, distances, fstate)

    return fstate.cost


def get_lowest_cost_ordering(subgoals, distances, do_sort=True):
    """This computes the lowest cost ordering (the policy) the robot will follow
    for navigation under uncertainty. It wraps a branch-and-bound search
    function implemented in C++ in 'lsp_accel'. As is typical of
    branch-and-bound functions, function evaluation is fastest if the high-cost
    plans can be ruled out quickly: i.e., if the first expansion is already of
    relatively low cost, many of the other branches can be pruned. When
    'do_sort' is True, a handful of common-sense heuristics are run to find an
    initial ordering that is of low cost to take advantage of this property. The
    subgoals are sorted by the various heuristics and the ordering that
    minimizes the expected cost is chosen. That ordering is used as an input to
    the search function, which searches it first."""

    if len(subgoals) == 0:
        return None, None

    if do_sort:
        order_heuristics = []
        order_heuristics.append({
            s: ii for ii, s in enumerate(subgoals)
        })
        order_heuristics.append({
            s: 1 - s.prob_feasible for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.prob_feasible * s.delta_success_cost +
            (1 - s.prob_feasible) * s.exploration_cost
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s]
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.delta_success_cost
            for s in subgoals
        })
        order_heuristics.append({
            s: distances['goal'][s] + distances['robot'][s] +
            s.exploration_cost
            for s in subgoals
        })

        heuristic_ordering_dat = []
        for heuristic in order_heuristics:
            ordered_subgoals = sorted(subgoals, reverse=False, key=lambda s: heuristic[s])
            ordering_cost = lsp.core.get_ordering_cost(ordered_subgoals, distances)
            heuristic_ordering_dat.append((ordering_cost, ordered_subgoals))

        subgoals = min(heuristic_ordering_dat, key=lambda hod: hod[0])[1]

    s_dict = {hash(s): s for s in subgoals}
    rd_cpp = {hash(s): distances['robot'][s] for s in subgoals}
    gd_cpp = {hash(s): distances['goal'][s] for s in subgoals}
    fd_cpp = {(hash(sp[0]), hash(sp[1])): distances['frontier'][frozenset(sp)]
              for sp in itertools.permutations(subgoals, 2)}
    s_cpp = [
        lsp_accel.FrontierData(s.prob_feasible, s.delta_success_cost,
                               s.exploration_cost, hash(s),
                               s.is_from_last_chosen) for s in subgoals
    ]

    cost, ordering = lsp_accel.get_lowest_cost_ordering(
        s_cpp, rd_cpp, gd_cpp, fd_cpp)
    ordering = [s_dict[sid] for sid in ordering]

    return cost, ordering


def get_lowest_cost_ordering_beginning_with(frontier_of_interest,
                                            subgoals,
                                            distances,
                                            do_sort=False):
    subgoals = [s for s in subgoals if not s == frontier_of_interest]

    if len(subgoals) == 0:
        state = FState(frontier_of_interest, distances)
        return state.cost, state.frontier_list
    if len(subgoals) == 1:
        state = FState(frontier_of_interest, distances)
        state = FState(subgoals[0], distances, state)
        return state.cost, state.frontier_list

    if frontier_of_interest not in subgoals:
        subgoals.append(frontier_of_interest)

    if do_sort:
        try:
            h = {
                s: distances['goal'][s] + distances['robot'][s] +
                s.prob_feasible * s.delta_success_cost +
                (1 - s.prob_feasible) * s.exploration_cost
                for s in subgoals
            }
        except KeyError:
            h = {
                s: distances['goal'][s.id] + distances['robot'][s.id] +
                s.prob_feasible * s.delta_success_cost +
                (1 - s.prob_feasible) * s.exploration_cost
                for s in subgoals
            }
        subgoals.sort(reverse=False, key=lambda s: h[s])

    s_dict = {hash(s): s for s in subgoals}
    try:
        rd_cpp = {hash(s): distances['robot'][s] for s in subgoals}
        gd_cpp = {hash(s): distances['goal'][s] for s in subgoals}
        fd_cpp = {(hash(sp[0]), hash(sp[1])):
                  distances['frontier'][frozenset(sp)]
                  for sp in itertools.permutations(subgoals, 2)}
    except KeyError:
        rd_cpp = {hash(s): distances['robot'][s.id] for s in subgoals}
        gd_cpp = {hash(s): distances['goal'][s.id] for s in subgoals}
        fd_cpp = {(hash(sp[0]), hash(sp[1])):
                  distances['frontier'][frozenset([sp[0].id, sp[1].id])]
                  for sp in itertools.permutations(subgoals, 2)}

    s_cpp = [
        lsp_accel.FrontierData(s.prob_feasible, s.delta_success_cost,
                               s.exploration_cost, hash(s),
                               s.is_from_last_chosen) for s in subgoals
        if s != frontier_of_interest
    ]

    foi = lsp_accel.FrontierData(frontier_of_interest.prob_feasible,
                                 frontier_of_interest.delta_success_cost,
                                 frontier_of_interest.exploration_cost,
                                 hash(frontier_of_interest),
                                 frontier_of_interest.is_from_last_chosen)

    cost, ordering = lsp_accel.get_lowest_cost_ordering_beginning_with(
        foi, s_cpp, rd_cpp, gd_cpp, fd_cpp)
    ordering = [s_dict[sid] for sid in ordering]

    return cost, ordering


def get_lowest_cost_ordering_not_beginning_with(frontier_of_interest,
                                                subgoals,
                                                distances,
                                                do_sort=False):
    subgoals = [s for s in subgoals if not s == frontier_of_interest]

    if len(subgoals) == 0:
        state = FState(frontier_of_interest, distances)
        return state.cost, state.frontier_list
    if len(subgoals) == 1:
        state = FState(frontier_of_interest, distances)
        state = FState(subgoals[0], distances, state)
        return state.cost, state.frontier_list

    if frontier_of_interest not in subgoals:
        subgoals.append(frontier_of_interest)

    if do_sort:
        try:
            h = {
                s: distances['goal'][s] + distances['robot'][s] +
                s.prob_feasible * s.delta_success_cost +
                (1 - s.prob_feasible) * s.exploration_cost
                for s in subgoals
            }
        except KeyError:
            h = {
                s: distances['goal'][s.id] + distances['robot'][s.id] +
                s.prob_feasible * s.delta_success_cost +
                (1 - s.prob_feasible) * s.exploration_cost
                for s in subgoals
            }
        subgoals.sort(reverse=False, key=lambda s: h[s])

    s_dict = {hash(s): s for s in subgoals}
    try:
        rd_cpp = {hash(s): distances['robot'][s] for s in subgoals}
        gd_cpp = {hash(s): distances['goal'][s] for s in subgoals}
        fd_cpp = {(hash(sp[0]), hash(sp[1])):
                  distances['frontier'][frozenset(sp)]
                  for sp in itertools.permutations(subgoals, 2)}
    except KeyError:
        rd_cpp = {hash(s): distances['robot'][s.id] for s in subgoals}
        gd_cpp = {hash(s): distances['goal'][s.id] for s in subgoals}
        fd_cpp = {(hash(sp[0]), hash(sp[1])):
                  distances['frontier'][frozenset([sp[0].id, sp[1].id])]
                  for sp in itertools.permutations(subgoals, 2)}

    s_cpp = [
        lsp_accel.FrontierData(s.prob_feasible, s.delta_success_cost,
                               s.exploration_cost, hash(s),
                               s.is_from_last_chosen) for s in subgoals
    ]

    foi = lsp_accel.FrontierData(frontier_of_interest.prob_feasible,
                                 frontier_of_interest.delta_success_cost,
                                 frontier_of_interest.exploration_cost,
                                 hash(frontier_of_interest),
                                 frontier_of_interest.is_from_last_chosen)

    cost, ordering = lsp_accel.get_lowest_cost_ordering_not_beginning_with(
        foi, s_cpp, rd_cpp, gd_cpp, fd_cpp)
    ordering = [s_dict[sid] for sid in ordering]

    return cost, ordering


def get_lowest_cost_ordering_old(frontiers, distances):
    """Recursively compute the lowest cost ordering of provided frontiers.
    """
    def get_ordering_sub(frontiers, state=None):
        """Sub-function defined for recursion. Property 'bound' is set for
        branch-and-bound, which vastly speeds up computation in practice."""
        if len(frontiers) == 1:
            s = FState(frontiers[0], distances, state)
            get_ordering_sub.bound = min(s.cost, get_ordering_sub.bound)
            return s

        if state is not None and state.cost > get_ordering_sub.bound:
            return state

        try:
            return min([
                get_ordering_sub([fn for fn in frontiers if fn != f],
                                 FState(f, distances, state))
                for f in frontiers
            ])
        except ValueError:
            return None

    get_ordering_sub.bound = 1e10
    h = {
        s: distances['goal'][s] + distances['robot'][s] +
        s.prob_feasible * s.delta_success_cost +
        (1 - s.prob_feasible) * s.exploration_cost
        for s in frontiers
    }
    frontiers.sort(reverse=False, key=lambda s: h[s])

    best_state = get_ordering_sub(frontiers)
    if best_state is None:
        return None, None
    else:
        return best_state.cost, best_state.frontier_list


def get_top_n_frontiers(frontiers, goal_dist, robot_dist, n):
    """This heuristic is for retrieving the 'best' N frontiers"""

    # This sorts the frontiers by (1) any frontiers that "derive their
    # properties" from the last chosen frontier and (2) the probablity that the
    # frontiers lead to the goal.
    frontiers = [f for f in frontiers if f.prob_feasible > 0]

    h_prob = {s: s.prob_feasible for s in frontiers}
    try:
        h_dist = {s: goal_dist[s] + robot_dist[s] for s in frontiers}
    except KeyError:
        h_dist = {s: goal_dist[s.id] + robot_dist[s.id] for s in frontiers}

    fs_prob = sorted(list(frontiers), key=lambda s: h_prob[s], reverse=True)
    fs_dist = sorted(list(frontiers), key=lambda s: h_dist[s], reverse=False)

    seen = set()
    fs_collated = []

    for front_d in fs_dist[:2]:
        if front_d not in seen:
            seen.add(front_d)
            fs_collated.append(front_d)

    for front_p in fs_prob:
        if front_p not in seen:
            seen.add(front_p)
            fs_collated.append(front_p)

    assert len(fs_collated) == len(seen)
    assert len(fs_collated) == len(fs_prob)
    assert len(fs_collated) == len(fs_dist)

    return fs_collated[0:n]


def get_top_n_frontiers_distance(frontiers, goal_dist, robot_dist, n):
    """This heuristic is for retrieving the 'best' N frontiers"""

    frontiers = [f for f in frontiers if f.prob_feasible > 0]

    try:
        h_dist = {s: goal_dist[s] + robot_dist[s] for s in frontiers}
    except KeyError:
        h_dist = {s: goal_dist[s.id] + robot_dist[s.id] for s in frontiers}

    fs_dist = sorted(list(frontiers), key=lambda s: h_dist[s], reverse=False)

    return fs_dist[0:n]


def get_best_expected_cost_and_frontier_list(grid,
                                             robot_pose,
                                             goal_pose,
                                             frontiers,
                                             num_frontiers_max=0,
                                             downsample_factor=1,
                                             do_correct_low_prob=False):
    """Compute the best frontier using the LSP algorithm."""
    logger = logging.getLogger("frontier")

    # Remove frontiers that are infeasible
    frontiers = [f for f in frontiers if f.prob_feasible != 0]

    # Calculate the distance to the goal, if infeasible, remove frontier
    stime = time.time()
    goal_distances = get_goal_distances(grid,
                                        goal_pose,
                                        frontiers=frontiers,
                                        downsample_factor=downsample_factor)
    frontiers = [f for f in frontiers if f.prob_feasible != 0]
    logger.debug(f"time to get goal_distances: {time.time() - stime}")

    stime = time.time()
    robot_distances = get_robot_distances(grid,
                                          robot_pose,
                                          frontiers=frontiers,
                                          downsample_factor=downsample_factor)
    logger.debug(f"time to get robot_distances: {time.time() - stime}")

    # Get the most n probable frontiers to limit computational load
    if num_frontiers_max > 0 and num_frontiers_max < len(frontiers):
        frontiers = get_top_n_frontiers(frontiers, goal_distances,
                                        robot_distances, num_frontiers_max)

    # Calculate robot and frontier distances
    stime = time.time()
    frontier_distances = get_frontier_distances(
        grid, frontiers=frontiers, downsample_factor=downsample_factor)
    logger.debug(f"time to get frontier_distances: {time.time() - stime}")

    # Make one last pass to eliminate infeasible frontiers
    frontiers = [f for f in frontiers if f.prob_feasible != 0]

    distances = {
        'frontier': frontier_distances,
        'robot': robot_distances,
        'goal': goal_distances,
    }

    stime = time.time()

    if do_correct_low_prob:
        old_probs = [f.prob_feasible for f in frontiers]
        sum_old_probs = sum(old_probs)
        if sum_old_probs < 1.0:
            # print("[WARN] Fixing low probability frontiers.")
            for f in frontiers:
                f.prob_feasible /= sum_old_probs

    out = get_lowest_cost_ordering(frontiers, distances)
    logger.debug(f"time to get ordering: {time.time() - stime}")

    if do_correct_low_prob and sum_old_probs < 1.0:
        for f in frontiers:
            f.prob_feasible *= sum_old_probs

    return out
