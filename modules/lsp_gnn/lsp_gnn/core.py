import lsp
import lsp_gnn
import numpy as np


IS_FROM_LAST_CHOSEN_REWARD = 1 * 10.0  # Fix me: See if it was really helping


class RolloutState(object):
    """Used to conviently store the 'state' during recursive cost search.
    """
    def __init__(self, subgoal_props, distances, frontier_idx, subgoals, old_state=None):
        if old_state is None:
            # create the history tuple
            self.history = [1] * len(list(subgoal_props.keys())[0])
        else:
            # update history in accordance with previous subgoal
            self.history = old_state.history.copy()
            self.history[old_state.frontier_list[-1]] = 0

        nf = subgoal_props[tuple(self.history)][frontier_idx]
        p = nf[0]  # set the prob_feasible of the frontier in p
        # Success cost
        sc = nf[1] + distances['goal'][frontier_idx]
        # Exploration cost
        ec = nf[2]

        if old_state is not None:
            self.frontier_list = old_state.frontier_list + [frontier_idx]
            # Store the old frontier
            of = old_state.frontier_list[-1]
            # Known cost (travel between frontiers)
            kc = distances['frontier'][frozenset([frontier_idx, of])]
            self.cost = old_state.cost + old_state.prob * (kc + p * sc +
                                                           (1 - p) * ec)
            self.prob = old_state.prob * (1 - p)
        else:
            # This is the first frontier, so the robot must accumulate a
            # cost of getting to the frontier
            self.frontier_list = [frontier_idx]
            # Known cost (travel to frontier)
            kc = distances['robot'][frontier_idx]
            if subgoals[frontier_idx].is_from_last_chosen:
                kc -= IS_FROM_LAST_CHOSEN_REWARD
            self.cost = kc + p * sc + (1 - p) * ec
            self.prob = (1 - p)

    def __lt__(self, other):
        return self.cost < other.cost


def get_lowest_cost_ordering(subgoal_props, distances, subgoals):
    """Recursively compute the lowest cost ordering of provided frontiers.
    """
    def get_ordering_sub(frontiers, state=None):
        """Sub-function defined for recursion. Property 'bound' is set for
        branch-and-bound, which vastly speeds up computation in practice."""
        if len(frontiers) == 1:
            s = RolloutState(
                subgoal_props=subgoal_props,
                distances=distances,
                frontier_idx=frontiers[0],
                subgoals=subgoals,
                old_state=state)
            get_ordering_sub.bound = min(s.cost, get_ordering_sub.bound)
            return s

        if state is not None and state.cost > get_ordering_sub.bound:
            return state

        try:
            return min([
                get_ordering_sub(
                    [fn for fn in frontiers if fn != f],
                    RolloutState(
                        subgoal_props=subgoal_props,
                        distances=distances,
                        frontier_idx=f,
                        subgoals=subgoals,
                        old_state=state
                    )
                )
                for f in frontiers
            ])
        except ValueError:
            return None

    length_of_frontiers = len(list(subgoal_props.keys())[0])
    # print('length_of_frontiers: ', length_of_frontiers)
    initial_history = tuple([1] * length_of_frontiers)
    frontiers = list(range(length_of_frontiers))
    get_ordering_sub.bound = 1e10
    h = {
        s: distances['goal'][s] + distances['robot'][s] +
        subgoal_props[initial_history][s][0] * subgoal_props[initial_history][s][1] +
        (1 - subgoal_props[initial_history][s][0]) * subgoal_props[initial_history][s][2]
        for s in frontiers
    }
    frontiers.sort(reverse=False, key=lambda s: h[s])

    best_state = get_ordering_sub(frontiers)
    if best_state is None:
        return None, None
    else:
        return best_state.cost, best_state.frontier_list


def get_new_conditional_data(subgoals, new_data, history):
    """ This method creates data later to be used by planning
    Takes input of
        1. The subgoals
        2. Data dictonary build only using the subgoal
        3. The history mapped from the history used in GCN
        i.e., if in gcn history was [0, 0, 1] where the first value is for
        a structural node and later two are for two subgoal nodes [s2, s1],
        then the history passed to planning is [1, 0] for the subgoal
        order [s1, s2]
    """
    conditional_subgoal_props = np.array([[subgoal.prob_feasible,
                                          subgoal.delta_success_cost,
                                          subgoal.exploration_cost]  # Nx3
                                          for subgoal in subgoals])
    new_data[tuple(history)] = conditional_subgoal_props


def convert_old_distance_to_new_distance(frontiers, old_distances):
    subgoal_ind_dict = {subgoal: ii for ii, subgoal in enumerate(frontiers)}
    new_distances = {}
    new_distances['goal'] = {
        subgoal_ind_dict[subgoal]: goal_dist
        for subgoal, goal_dist in old_distances['goal'].items()
        if subgoal in frontiers
    }
    new_distances['robot'] = {
        subgoal_ind_dict[subgoal]: robot_dist
        for subgoal, robot_dist in old_distances['robot'].items()
        if subgoal in frontiers
    }
    if old_distances['frontier']:   # In case there is only one frontier
        new_distances['frontier'] = {
            frozenset([subgoal_ind_dict[s] for s in old_key if s in frontiers]): subgoal_dist
            for old_key, subgoal_dist in old_distances['frontier'].items()}
    return new_distances


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


def get_best_expected_cost_and_frontier_list(grid,
                                             robot_pose,
                                             goal_pose,
                                             frontiers,
                                             vertex_points,
                                             subgoal_nodes,
                                             gcn_graph_input,
                                             subgoal_property_net,
                                             num_frontiers_max=0,
                                             downsample_factor=1):
    """Compute the best frontier using the LSP algorithm."""

    initial_frontiers = [f for f in frontiers]  # Change it to all frontiers
    # Remove frontiers that are infeasible
    frontiers = [f for f in frontiers if f.prob_feasible != 0]

    # Calculate the distance to the goal, if infeasible, remove frontier
    goal_distances = lsp.core.get_goal_distances(
        grid, goal_pose, frontiers=frontiers,
        downsample_factor=downsample_factor
    )
    frontiers = [f for f in frontiers if f.prob_feasible != 0]

    robot_distances = lsp.core.get_robot_distances(
        grid, robot_pose, frontiers=frontiers,
        downsample_factor=downsample_factor
    )
    # Get the most n probable frontiers to limit computational load
    if num_frontiers_max > 0 and num_frontiers_max < len(frontiers):
        # print("Took top 8 frontiers")
        top_frontiers = get_top_n_frontiers(frontiers, goal_distances,
                                            robot_distances, num_frontiers_max)
        frontiers = [f for f in frontiers if f in top_frontiers]

    # Calculate robot and frontier distances
    frontier_distances = lsp.core.get_frontier_distances(
        grid, frontiers=frontiers, downsample_factor=downsample_factor)

    # Make one last pass to eliminate infeasible frontiers
    frontiers = [f for f in frontiers if f.prob_feasible != 0]
    old_distances = {
        'frontier': frontier_distances,
        'robot': robot_distances,
        'goal': goal_distances,
    }
    distances = convert_old_distance_to_new_distance(frontiers, old_distances)

    # Fix the 'history' vector for the pruned subgoal before generating
    # the rollout histories
    # vertex_count = len(gcn_graph_input['is_subgoal'])
    new_initial_history = gcn_graph_input['history'].copy()
    subgoals = list(subgoal_nodes.values())
    subgoal_vertices = list(subgoal_nodes.keys())
    list_of_vertex_points = vertex_points.tolist()
    for f in initial_frontiers:
        if f not in frontiers:
            lookup_vertex = list(subgoal_vertices[subgoals.index(f)])
            index = list_of_vertex_points.index(lookup_vertex)
            new_initial_history[index] = 0

    # Estimate properties of all subgoals for all history rollout histories
    new_data = {}
    # Need to calculate the histories first
    all_history = lsp_gnn.utils. \
        generate_all_rollout_history(new_initial_history)

    for history in all_history:
        gcn_graph_input['history'] = history
        prob_feasible_dict, dsc, ec, _ = subgoal_property_net(
            datum=gcn_graph_input,
            vertex_points=vertex_points,
            subgoals=frontiers)

        # Creating history for new data
        history_for_new_data_key = []
        for subgoal in frontiers:
            possible_node = lsp_gnn.utils. \
                get_subgoal_node(vertex_points, subgoal).tolist()
            index = list_of_vertex_points.index(possible_node)
            history_for_new_data_key.append(
                gcn_graph_input['history'][index]
            )
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal],
                delta_success_cost=dsc[subgoal],
                exploration_cost=ec[subgoal])
        lsp_gnn.core.get_new_conditional_data(
            subgoals=frontiers,
            new_data=new_data,
            history=history_for_new_data_key
        )

    conditional_out = lsp_gnn.core.get_lowest_cost_ordering(new_data, distances, frontiers)
    # return conditional_out
    return (conditional_out[0],
            [initial_frontiers.index(frontiers[co_ind])
            for co_ind in conditional_out[1]])
