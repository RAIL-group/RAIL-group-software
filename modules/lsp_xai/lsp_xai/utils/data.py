import lsp


def _update_datum_single_policy(subgoal_props,
                                distances,
                                relevant_subgoal_inds,
                                target_subgoal_ind,
                                beginning_with=True):
    """Helper function for update_datum_policies."""

    selected_subgoal = subgoal_props[target_subgoal_ind]

    # Recompute the policy (using old policy as starting point)
    subgoals = [subgoal_props[inds] for inds in relevant_subgoal_inds]
    if beginning_with:
        policy_sg = lsp.core.get_lowest_cost_ordering_beginning_with(
            selected_subgoal, subgoals, distances, do_sort=True)[1]
    else:
        # Get the policy starting with the second-best subgoal
        policy_sg = lsp.core.get_lowest_cost_ordering_not_beginning_with(
            selected_subgoal, subgoals, distances, do_sort=True)[1]

    policy_inds = [s.id for s in policy_sg]

    # Compute the distances vectors
    robot_distance = distances['robot'][policy_inds[0]]
    success_distances = []
    failure_distances = []

    # Loop through all concecutive pairs of frontiers
    # specified by the policy and populate the distance
    # arrays.
    for fprev, fnext in zip(policy_inds, policy_inds[1:]):
        success_distances.append(distances['goal'][fprev])
        failure_distances.append(distances['frontier'][frozenset(
            [fprev, fnext])])

    # Handle the final frontier
    success_distances.append(distances['goal'][policy_inds[-1]])
    failure_distances.append(0)

    # Return a dictionary containing the data
    return {
        'policy': policy_inds,
        'robot_distance': robot_distance,
        'success_distances': success_distances,
        'failure_distances': failure_distances
    }


def update_datum_policies(subgoal_props, datum):
    """Using the provided subgoal_props, recompute the subgoal ordering beginning
with the same subgoal that would minimize the expected cost. The datum itself is
updated and returned, as other data (in particular the distances) need to be
updated."""
    target_subgoal_ind = int(datum['target_subgoal_ind'])
    if 'backup_subgoal_ind' in datum.keys():
        backup_subgoal_ind = datum['backup_subgoal_ind']
    else:
        backup_subgoal_ind = None

    all_subgoals = [
        s for s in list(subgoal_props.values())
        if datum['distances']['goal'][s.id] < 1e8
        and datum['distances']['robot'][s.id] < 1e8
    ]
    if target_subgoal_ind not in [s.id for s in all_subgoals]:
        return None

    relevant_subgoal_inds = [
        s.id for s in lsp.core.get_top_n_frontiers(
            all_subgoals,
            datum['distances']['goal'],
            datum['distances']['robot'],
            n=lsp.planners.subgoal_planner.NUM_MAX_FRONTIERS)
    ]

    # Ensure the citical subgoals are in the final set
    if target_subgoal_ind not in relevant_subgoal_inds:
        relevant_subgoal_inds.append(target_subgoal_ind)
    if backup_subgoal_ind is not None:
        if backup_subgoal_ind not in relevant_subgoal_inds:
            relevant_subgoal_inds.append(backup_subgoal_ind)

    datum['target_subgoal_policy'] = _update_datum_single_policy(
        subgoal_props,
        datum['distances'],
        relevant_subgoal_inds,
        target_subgoal_ind,
        beginning_with=True)
    if backup_subgoal_ind is not None:
        datum['backup_subgoal_policy'] = _update_datum_single_policy(
            subgoal_props,
            datum['distances'],
            relevant_subgoal_inds,
            backup_subgoal_ind,
            beginning_with=True)
    else:
        datum['backup_subgoal_policy'] = _update_datum_single_policy(
            subgoal_props,
            datum['distances'],
            relevant_subgoal_inds,
            target_subgoal_ind,
            beginning_with=False)
    return datum


def compute_expected_cost_for_policy(subgoal_props, subgoal_policy_data):

    expected_cost = subgoal_policy_data['robot_distance']
    success_distances = subgoal_policy_data['success_distances']
    failure_distances = subgoal_policy_data['failure_distances']

    ordered_subgoal_props = [
        subgoal_props[int(ind)] for ind in subgoal_policy_data['policy']
    ]

    expected_costs = [subgoal_policy_data['robot_distance']]
    failure_probs = [1.0, 1.0]
    for f, sd, fd in zip(ordered_subgoal_props, success_distances,
                         failure_distances):
        sprob = f.prob_feasible
        fprob = (1 - sprob)
        fpXsc = sprob * (sd + f.delta_success_cost)
        fnpXec = fprob * (f.exploration_cost + fd)
        expected_costs.append(fpXsc + fnpXec)
        failure_probs.append(failure_probs[-1] * fprob)

    expected_cost = sum(
        [ec * fp for ec, fp in zip(expected_costs, failure_probs)])

    return expected_cost
