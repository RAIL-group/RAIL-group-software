"""Tests for interpretability project."""

import environments
import learning
import lsp
import lsp_xai
import os
import pytest
import tempfile
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def guided_maze_planner(unity_path, lsp_xai_maze_net_0SG_path):
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args(['--save_dir', ''])
    args.current_seed = 1037
    args.map_type = 'maze'
    args.step_size = 1.8
    args.num_primitives = 16
    args.field_of_view_deg = 360
    args.base_resolution = 1.0
    args.inflation_radius_m = 2.5
    args.laser_max_range_m = 60
    args.unity_path = unity_path
    args.num_range = 32
    args.num_bearing = 128
    args.network_file = lsp_xai_maze_net_0SG_path

    # Create the map
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
    builder = environments.simulated.WorldBuildingUnityBridge

    # Helper function for creating a new robot instance
    def get_initialized_robot():
        return lsp.robot.Turtlebot_Robot(pose,
                                         primitive_length=args.step_size,
                                         num_primitives=args.num_primitives,
                                         map_data=map_data)

    with builder(args.unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        return lsp_xai.planners.evaluate.run_model_eval(
            args,
            goal,
            known_map,
            simulator,
            unity_bridge,
            get_initialized_robot(),
            eval_planner=lsp_xai.planners.SubgoalPlanner(goal=goal, args=args),
            known_planner=None,
            return_planner_after_steps=20)


def test_compute_backup_subgoal(guided_maze_planner):
    """I want to compute the backup subgoal: the subgoal the agent should choose if
it's told that its first choice is unavailable. To do this, I should confirm
that the function 'get_lowest_cost_ordering_not_beginning_with' works as
intended. I will loop through all subgoals and use
'get_lowest_cost_ordering_beginning_with' to compute the best ordering beginning
with each of the subgoals. I should confirm that the target subgoal is the one
with the lowest cost and that the backup subgoal is the one with the second
lowest cost."""
    planner = guided_maze_planner
    subgoals, distances = planner.get_subgoals_and_distances()

    cost_ordering_list = []
    for subgoal in subgoals:
        ordering_data = lsp.core.get_lowest_cost_ordering_beginning_with(
            subgoal, subgoals, distances)
        assert len(subgoals) == len(ordering_data[1])
        assert subgoal == ordering_data[1][0]
        cost_ordering_list.append(ordering_data)

    # Confirm that getting the ordering *not* beginning with subgoal works
    print(f"num subgoals: {len(subgoals)}")
    for subgoal in subgoals:
        ordering_data = lsp.core.get_lowest_cost_ordering_not_beginning_with(
            subgoal, subgoals, distances)
        assert not subgoal == ordering_data[1][0]
        assert subgoal in ordering_data[1]
        assert len(subgoals) == len(ordering_data[1])

    selected_subgoal = planner.compute_selected_subgoal()
    backup_subgoal = planner.compute_backup_subgoal(selected_subgoal)

    cost_ordering_list.sort(key=lambda co: co[0])

    print(cost_ordering_list[0][1])
    print(cost_ordering_list[1][1])
    print(backup_subgoal)
    assert selected_subgoal == cost_ordering_list[0][1][0]
    assert backup_subgoal == cost_ordering_list[1][1][0]


@pytest.mark.parametrize("policy_name",
                         ["target_subgoal_policy", "backup_subgoal_policy"])
def test_compute_policy_during_training(guided_maze_planner, policy_name):
    """Confirm that the policy itself is not stored in the training data and that it
can be computed from the training datum itself. Elsewhere, I confirm that the
policy is computed correctly; this is only to confirm that it is computed at
all."""
    # Rename planner for convenience
    planner = guided_maze_planner

    # Compute the original datum
    selected_subgoal = planner.compute_selected_subgoal()
    datum_raw = planner.compute_subgoal_data(selected_subgoal)
    assert policy_name not in datum_raw.keys()
    assert 'target_subgoal_ind' in datum_raw.keys()

    # Compute the updated datum with the new properties and policy data
    datum_updated = planner.model.update_datum(datum_raw, device=DEVICE)
    assert policy_name in datum_updated.keys()
    assert 'policy' in datum_updated[policy_name]
    assert 'robot_distance' in datum_updated[policy_name]
    assert 'success_distances' in datum_updated[policy_name]
    assert 'failure_distances' in datum_updated[policy_name]


def test_backup_ind_determines_policy(guided_maze_planner):
    """Confirm that when 'backup_subgoal_ind' is set, that this ind determines which
subgoal is the first subgoal in the 'backup_subgoal_policy'."""
    # Rename planner for convenience
    planner = guided_maze_planner

    # Compute the datum and update
    selected_subgoal = planner.compute_selected_subgoal()
    datum_raw = planner.compute_subgoal_data(selected_subgoal)
    datum = planner.model.update_datum(datum_raw, device=DEVICE)
    target_policy = datum['target_subgoal_policy']['policy']

    # Now mix up the backup subgoal ind a bit
    for new_backup_ind in target_policy[:5]:
        datum['backup_subgoal_ind'] = new_backup_ind
        datum = planner.model.update_datum(datum, device=DEVICE)
        assert new_backup_ind == datum['backup_subgoal_policy']['policy'][0]


def test_policy_tree_data_policies_match(guided_maze_planner):
    """Regression test. This test confirms that the two policies have the same
subgoals in them. This test was created after a bug was identified that the
'backup' policy did not contain the 'target' subgoal."""
    # Rename the planner for convenience
    planner = guided_maze_planner

    # Compute and "update" (initialize) the datum
    selected_subgoal = planner.compute_selected_subgoal()
    policy_data = planner.compute_subgoal_data(selected_subgoal)
    datum = planner.model.update_datum(policy_data, device=DEVICE)

    # Confirm that they contain the same subgoals
    assert (set(datum['target_subgoal_policy']['policy']) == set(
        datum['backup_subgoal_policy']['policy']))

    # Confirm that the do not begin with the same subgoals
    assert not (datum['target_subgoal_policy']['policy'][0]
                == datum['backup_subgoal_policy']['policy'][0])


def test_compute_expected_cost(guided_maze_planner):
    """The purpose of this test is to confirm that the expected cost computed by
    PyTorch is the same as the expected cost computed by the frontier.py
    functions."""
    # Get the planner, subgoals, and distances
    planner = guided_maze_planner
    subgoals, distances = planner.get_subgoals_and_distances()
    selected_subgoal = planner.compute_selected_subgoal()

    # Compute the cost via frontier.py
    f_cost, f_order = lsp.core.get_lowest_cost_ordering_beginning_with(
        selected_subgoal, subgoals, distances)
    print("cost, order", f_cost, f_order)
    assert f_order[0] == selected_subgoal

    # === Compute the cost via PyTorch ===
    # Compute the datum
    device = DEVICE
    policy_data = planner.compute_subgoal_data(selected_subgoal)
    datum = planner.model.update_datum(policy_data, device=DEVICE)

    with torch.no_grad():
        # Compute Subgoal Properties
        out, ind_map = planner.model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(out[:, 0])
        delta_success_costs = out[:, 1]
        exploration_costs = out[:, 2]
        subgoal_props, _, _ = planner.model.compute_subgoal_props(
            is_feasibles, delta_success_costs, exploration_costs,
            datum['subgoal_data'], ind_map, device)

        # Compute the cost
        pt_cost = planner.model.compute_expected_cost_for_policy(
            subgoal_props, datum['target_subgoal_policy'])

    print("F order: ", [s.prob_feasible for s in f_order])
    print(f"F cost: {f_cost}")
    print("PT order: ", [
        subgoal_props[ind].prob_feasible
        for ind in datum['target_subgoal_policy']['policy']
    ])
    print(f"PT cost: {pt_cost}")

    assert abs(pt_cost - f_cost) < 0.5


def test_compute_parameter_deltas_no_change(guided_maze_planner):
    """Provided two subgoals (presumably 'target' and 'backup') the system should
use PyTorch to generate a gradient signal for the subgoal properties. This test
is to verify that computing the subgoal properties does not alter the trained
model. """
    # Get the planner, subgoals, and distances
    planner = guided_maze_planner
    subgoals, distances = planner.get_subgoals_and_distances()
    selected_subgoal = planner.compute_selected_subgoal()

    device = DEVICE
    model = planner.model

    # Get and convert the data
    policy_data = planner.compute_subgoal_data(selected_subgoal)
    datum = planner.model.update_datum(policy_data, device=DEVICE)

    with torch.no_grad():
        out, ind_map = model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(out[:, 0])
        delta_success_costs = out[:, 1]
        exploration_costs = out[:, 2]
        subgoal_props_base, _, _ = model.compute_subgoal_props(
            is_feasibles, delta_success_costs, exploration_costs,
            datum['subgoal_data'], ind_map, device)

    # Run the function to compute subgoal property impact
    model.get_subgoal_prop_impact(datum, device)

    # Compute reverted subgoal properties
    with torch.no_grad():
        out, ind_map = model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(out[:, 0])
        delta_success_costs = out[:, 1]
        exploration_costs = out[:, 2]
        subgoal_props_after, _, _ = model.compute_subgoal_props(
            is_feasibles, delta_success_costs, exploration_costs,
            datum['subgoal_data'], ind_map, device)

    # Confirm that the subgoals are unchanged before and after computing
    # the 'impact'.
    for ind in subgoal_props_base.keys():
        sb = subgoal_props_base[ind]
        sa = subgoal_props_after[ind]

        assert abs(sb.prob_feasible - sa.prob_feasible) < 1e-4
        assert abs(sb.delta_success_cost - sa.delta_success_cost) < 1e-4
        assert abs(sb.exploration_cost - sa.exploration_cost) < 1e-4


def test_compute_parameter_deltas_values(guided_maze_planner):
    """Provided two subgoals (presumably 'target' and 'backup') the system should
use PyTorch to generate a gradient signal for the subgoal properties. This test
will confirm that updating the parameters to use the new values results in the
change we expect."""
    # Get the planner, subgoals, and distances
    planner = guided_maze_planner
    subgoals, distances = planner.get_subgoals_and_distances()
    selected_subgoal = planner.compute_selected_subgoal()

    device = DEVICE
    model = planner.model

    # Get and convert the data
    policy_data = planner.compute_subgoal_data(selected_subgoal)
    datum = planner.model.update_datum(policy_data, device=DEVICE)

    with torch.no_grad():
        out, ind_map = model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(out[:, 0])
        delta_success_costs = out[:, 1]
        exploration_costs = out[:, 2]
        subgoal_props_base, _, _ = model.compute_subgoal_props(
            is_feasibles, delta_success_costs, exploration_costs,
            datum['subgoal_data'], ind_map, device)

    # Run the function
    subgoal_prop_update_data = model.get_subgoal_prop_impact(datum, device)

    print('target policy:')
    print(datum['target_subgoal_policy']['policy'])
    print('backup policy:')
    print(datum['backup_subgoal_policy']['policy'])
    for value in subgoal_prop_update_data.values():
        print(value)

    # Confirm that the 'rank' of the target and backup subgoals' prob_feasible
    # is in the top 6.
    assert subgoal_prop_update_data[(
        datum['target_subgoal_policy']['policy'][0], 'prob_feasible')].rank < 6
    assert subgoal_prop_update_data[(
        datum['backup_subgoal_policy']['policy'][0], 'prob_feasible')].rank < 6


@pytest.mark.parametrize("num_subgoals,other_costs", [(0, False), (0, True),
                                                      (1, False), (4, False),
                                                      (-1, False),
                                                      (100, False)])
def test_compute_parameter_deltas_train(guided_maze_planner, num_subgoals,
                                        other_costs):
    """Provided two subgoals (presumably 'target' and 'backup') the system should
use PyTorch to generate a gradient signal for the subgoal properties. This test
is to verify that computing the subgoal properties does not alter the trained
model. """
    # Get the planner, subgoals, and distances
    planner = guided_maze_planner
    subgoals, distances = planner.get_subgoals_and_distances()
    selected_subgoal = planner.compute_selected_subgoal()

    device = DEVICE
    model = planner.model

    # Get and convert the data
    policy_data = planner.compute_subgoal_data(selected_subgoal)
    datum = planner.model.update_datum(policy_data, device=DEVICE)
    datum['net_cost_remaining'] = 100
    datum['net_cost_remaining_known'] = 100

    with torch.no_grad():
        out, ind_map = model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(out[:, 0])
        delta_success_costs = out[:, 1]
        exploration_costs = out[:, 2]
        subgoal_props_base, _, _ = model.compute_subgoal_props(
            is_feasibles, delta_success_costs, exploration_costs,
            datum['subgoal_data'], ind_map, device)

    # Run the optimizer and train
    delta_subgoal_data = model.get_subgoal_prop_impact(datum, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    out, ind_map = model(datum, device)
    loss = model.loss(out,
                      datum,
                      ind_map,
                      device=device,
                      limit_subgoals_num=num_subgoals,
                      delta_subgoal_data=delta_subgoal_data,
                      do_include_limit_costs=other_costs,
                      do_include_negative_costs=other_costs)
    optimizer.zero_grad()
    if num_subgoals == 0 and not other_costs:
        with pytest.raises(RuntimeError):
            loss.backward()
        return
    loss.backward()
    optimizer.step()

    # Compute reverted subgoal properties
    with torch.no_grad():
        out, ind_map = model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(out[:, 0])
        delta_success_costs = out[:, 1]
        exploration_costs = out[:, 2]
        subgoal_props_after, _, _ = model.compute_subgoal_props(
            is_feasibles, delta_success_costs, exploration_costs,
            datum['subgoal_data'], ind_map, device)

    # Confirm that the subgoals are unchanged after updating the gradient with
    # no tensors.
    count = 0
    for ind in subgoal_props_base.keys():
        sb = subgoal_props_base[ind]
        sa = subgoal_props_after[ind]

        if abs(sb.prob_feasible - sa.prob_feasible) > 1e-4:
            count += 1
        if abs(sb.delta_success_cost - sa.delta_success_cost) > 1e-4:
            count += 1
        if abs(sb.exploration_cost - sa.exploration_cost) < 1e-4:
            count += 1

    assert count >= 1


@pytest.mark.skip(
    reason="slight API change; direct comparison no longer recommended.")
@pytest.mark.parametrize("new_target_ind", [0, 1])
def test_generate_explanation_partial_subgoals(guided_maze_planner,
                                               new_target_ind):
    """Generate an explanation by running gradient descent over the difference
between two policies: the target subgoal and the backup subgoal."""
    # Rename the planner for convenience
    planner = guided_maze_planner

    # Initialize the datum
    device = DEVICE
    chosen_subgoal = planner.compute_selected_subgoal()
    datum, subgoal_ind_dict = planner.compute_subgoal_data(
        chosen_subgoal, 24, do_return_ind_dict=True)
    datum = planner.model.update_datum(datum, device)

    # Confirm that the first subgoal of the computed policy corresponds to the
    # index associated with the chosen subgoal, a confirmation that the mapping
    # is formatted correctly.
    assert (subgoal_ind_dict[chosen_subgoal] == datum['target_subgoal_policy']
            ['policy'][0])

    # Now we want to rearrange things a bit: the new 'target' subgoal we set to
    # some other subgoal (the `new_target_ind`) and we populate the 'backup'
    # subgoal with the 'chosen' subgoal (the subgoal the agent actually chose).
    query_subgoal_ind = datum['backup_subgoal_policy']['policy'][
        new_target_ind]
    datum['target_subgoal_ind'] = query_subgoal_ind
    datum['backup_subgoal_ind'] = subgoal_ind_dict[chosen_subgoal]
    print(datum['target_subgoal_ind'])
    print(datum['backup_subgoal_ind'])

    # We update the datum to reflect this change (and confirm it worked).
    datum = planner.model.update_datum(datum, device)
    import copy
    assert (datum['target_subgoal_policy']['policy'][0] ==
            datum['target_subgoal_ind'])
    assert (datum['backup_subgoal_policy']['policy'][0] ==
            datum['backup_subgoal_ind'])

    # Compute the 'delta subgoal data'. This is how we determine the
    # 'importance' of each of the subgoal properties. In practice, we will sever
    # the gradient for all but a handful of these with an optional parameter
    # (not set here).
    delta_subgoal_data = planner.model.get_subgoal_prop_impact(
        datum, device, delta_cost_limit=-1e10)
    base_model_state = planner.model.state_dict(keep_vars=False)
    base_model_state = {k: v.cpu() for k, v in base_model_state.items()}
    base_model_state = copy.deepcopy(base_model_state)

    # Initialize some terms for the optimization
    learning_rate = 1.0e-4
    optimizer = torch.optim.SGD(planner.model.parameters(), lr=learning_rate)

    # Now we perfrom iterative gradient descent until the expected cost of the
    # new target subgoal is lower than that of the originally selected subgoal.
    for ii in range(5000):
        # Update datum to reflect new neural network state
        datum = planner.model.update_datum(datum, device)

        # Compute the subgoal properties by passing images through the network.
        # (PyTorch implicitly builds a graph of these operations so that we can
        # differentiate them later.)
        nn_out, ind_mapping = planner.model(datum, device)
        is_feasibles = torch.nn.Sigmoid()(nn_out[:, 0])
        delta_success_costs = nn_out[:, 1]
        exploration_costs = nn_out[:, 2]
        limited_subgoal_props, _, _ = planner.model.compute_subgoal_props(
            is_feasibles,
            delta_success_costs,
            exploration_costs,
            datum['subgoal_data'],
            ind_mapping,
            device,
            limit_subgoals_num=-1,  # We do not actually limit for this test
            delta_subgoal_data=delta_subgoal_data)

        print("")
        print(datum['target_subgoal_policy']['policy'])
        print(datum['backup_subgoal_policy']['policy'])

        # Compute the expected of the new target subgoal:
        q_target = planner.model.compute_expected_cost_for_policy(
            limited_subgoal_props, datum['target_subgoal_policy'])
        # Cost of the 'backup' (formerly the agent's chosen subgoal):
        q_backup = planner.model.compute_expected_cost_for_policy(
            limited_subgoal_props, datum['backup_subgoal_policy'])
        print(f"Q_target = {q_target} | Q_backup = {q_backup}")

        if ii == 0:
            # Store the original values for each.
            base_subgoal_props = limited_subgoal_props
            q_target_original = q_target.item()
            q_backup_original = q_backup.item()

        print(
            f"Q_target_o = {q_target_original} | Q_backup_o = {q_backup_original}"
        )
        # The zero-crossing of the difference between the two is the decision
        # boundary we are hoping to cross by updating the paramters of the
        # neural network via gradient descent.
        q_diff = q_target - q_backup

        if q_diff <= 0:
            # When it's less than zero, we're done.
            break

        # Via PyTorch magic, gradient descent is easy:
        optimizer.zero_grad()
        q_diff.backward()
        optimizer.step()
    else:
        # If it never crossed the boundary, we have failed.
        raise ValueError("Decision boundary never crossed.")

    # Compute the final subgoal property values
    upd_subgoal_props = limited_subgoal_props

    # Reload the model's state
    planner.model.load_state_dict(base_model_state)
    planner.model.eval()
    planner.model = planner.model.to(device)

    # Compute the explanation via the built-in function
    query_subgoal = [
        s for s, ind in subgoal_ind_dict.items() if ind == query_subgoal_ind
    ][0]
    explanation = planner.generate_counterfactual_explanation(query_subgoal)
    print(explanation)
    xai_prop_changes = explanation.get_subgoal_prop_changes()

    # Conf
    for ind in subgoal_ind_dict.values():
        if ind not in base_subgoal_props.keys():
            continue
        if ind not in upd_subgoal_props.keys():
            continue

        print(f"Ind: {ind}")
        print(f"  Old: Ps={base_subgoal_props[ind].prob_feasible:0.4f}")
        print(f"  New: Ps={upd_subgoal_props[ind].prob_feasible:0.4f}")

        if delta_subgoal_data[(ind, 'prob_feasible')].rank < 9:
            prob_diff = upd_subgoal_props[
                ind].prob_feasible - base_subgoal_props[ind].prob_feasible
            assert xai_prop_changes[ind].prob_feasible_diff == pytest.approx(
                prob_diff.item(), 5e-2)
        if delta_subgoal_data[(ind, 'delta_success_cost')].rank < 9:
            scost_diff = upd_subgoal_props[
                ind].delta_success_cost - base_subgoal_props[
                    ind].delta_success_cost
            assert xai_prop_changes[
                ind].delta_success_cost_diff == pytest.approx(
                    scost_diff.item(), 5e-2)


@pytest.mark.parametrize("new_target_ind", [1])
def test_explanation_generation_leaves_model_unchanged(guided_maze_planner,
                                                       new_target_ind):
    """Generate an explanation by running gradient descent over the difference
between two policies: the target subgoal and the backup subgoal."""
    # Rename the planner for convenience
    planner = guided_maze_planner

    # Datum needed here to compute 'query subgoal'.
    # Not necessary in general.
    device = DEVICE
    chosen_subgoal = planner.compute_selected_subgoal()
    datum, subgoal_ind_dict = planner.compute_subgoal_data(
        chosen_subgoal, 24, do_return_ind_dict=True)
    datum = planner.model.update_datum(datum, device)
    query_subgoal_ind = datum['backup_subgoal_policy']['policy'][
        new_target_ind]
    query_subgoal = [
        s for s, ind in subgoal_ind_dict.items() if ind == query_subgoal_ind
    ][0]

    # Generate the explanation twice
    explanation = planner.generate_counterfactual_explanation(query_subgoal, learning_rate=1e-5)
    xai_prop_changes_1 = explanation.get_subgoal_prop_changes()
    explanation = planner.generate_counterfactual_explanation(query_subgoal, learning_rate=1e-5)
    xai_prop_changes_2 = explanation.get_subgoal_prop_changes()

    for ind in xai_prop_changes_1.keys():
        assert xai_prop_changes_1[ind].prob_feasible_diff == pytest.approx(
            xai_prop_changes_2[ind].prob_feasible_diff, rel=0.05, abs=0.02)

    # The new chosen subgoal should be the same as the old
    new_chosen_subgoal = planner.compute_selected_subgoal()
    assert new_chosen_subgoal == chosen_subgoal


@pytest.mark.parametrize("new_target_ind", [1, 2])
def test_explanation_intervention_can_change_model(guided_maze_planner,
                                                   new_target_ind):
    """Generate an explanation but 'keep the changes' as in an intervention
experiment. The chosen subgoal after the explanations is generated should match
the query."""
    # Rename the planner for convenience
    planner = guided_maze_planner

    # Datum needed here to compute 'query subgoal'.
    # Not necessary in general.
    device = DEVICE
    chosen_subgoal = planner.compute_selected_subgoal()
    datum, subgoal_ind_dict = planner.compute_subgoal_data(
        chosen_subgoal, 24, do_return_ind_dict=True)
    datum = planner.model.update_datum(datum, device)
    query_subgoal_ind = datum['backup_subgoal_policy']['policy'][
        new_target_ind]
    query_subgoal = [
        s for s, ind in subgoal_ind_dict.items() if ind == query_subgoal_ind
    ][0]

    planner.generate_counterfactual_explanation(query_subgoal,
                                                do_freeze_selected=False,
                                                keep_changes=True,
                                                margin=2.0)
    print(query_subgoal)
    print(chosen_subgoal)

    # The new chosen subgoal should match the query subgoal
    assert planner.compute_selected_subgoal() == query_subgoal


@pytest.mark.parametrize("new_target_ind,limit_num", [(1, 4)])
def test_explanation_generation_succeeds_limit(guided_maze_planner,
                                               new_target_ind, limit_num):
    """Generate an explanation by running gradient descent over the difference
between two policies: the target subgoal and the backup subgoal."""
    # Rename the planner for convenience
    planner = guided_maze_planner

    # Datum needed here to compute 'query subgoal'.
    # Not necessary in general.
    device = DEVICE
    chosen_subgoal = planner.compute_selected_subgoal()
    datum, subgoal_ind_dict = planner.compute_subgoal_data(
        chosen_subgoal, 24, do_return_ind_dict=True)
    datum = planner.model.update_datum(datum, device)
    query_subgoal_ind = datum['target_subgoal_policy']['policy'][
        new_target_ind]
    query_subgoal = [
        s for s, ind in subgoal_ind_dict.items() if ind == query_subgoal_ind
    ][0]

    if limit_num == 0:
        # If no subgoal properties allowed, it should fail.
        with pytest.raises(RuntimeError):
            planner.generate_counterfactual_explanation(
                query_subgoal, limit_num)
    else:
        # Nothing to test except that it does not fail.
        planner.generate_counterfactual_explanation(query_subgoal,
                                                    limit_num,
                                                    do_freeze_selected=False)


@pytest.mark.parametrize("new_target_ind", [1, 2])
def test_explanation_can_visualize(do_debug_plot, guided_maze_planner,
                                   new_target_ind):
    """Generate and visualize an explanation (without crashing)."""
    # Rename the planner for convenience
    planner = guided_maze_planner

    # Datum needed here to compute 'query subgoal'.
    # Not necessary in general.
    device = DEVICE
    chosen_subgoal = planner.compute_selected_subgoal()
    datum, subgoal_ind_dict = planner.compute_subgoal_data(
        chosen_subgoal, 24, do_return_ind_dict=True)
    datum = planner.model.update_datum(datum, device)
    query_subgoal_ind = datum['backup_subgoal_policy']['policy'][
        new_target_ind]
    query_subgoal = [
        s for s, ind in subgoal_ind_dict.items() if ind == query_subgoal_ind
    ][0]

    explanation = planner.generate_counterfactual_explanation(query_subgoal)
    explanation.visualize(show_plot=do_debug_plot)


def test_save_load_subgoal_planner(guided_maze_planner):
    """Save planner data to a temporary file and load it back."""

    with tempfile.TemporaryDirectory() as pickle_dir:
        pickle_filename = os.path.join(pickle_dir, 'a_pickle.pickle.gz')

        datum = guided_maze_planner.get_planner_state()
        learning.data.write_compressed_pickle(pickle_filename, datum)
        datum_loaded = learning.data.load_compressed_pickle(pickle_filename)

        _ = lsp_xai.planners.SubgoalPlanner.create_with_state(datum_loaded, None)
