import environments
import lsp
import numpy as np
import pytest


def _get_env_and_args():
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args(['--save_dir', ''])
    args.current_seed = 1037
    args.map_type = 'maze'
    args.step_size = 1.8
    args.num_primitives = 16
    args.field_of_view_deg = 360
    args.base_resolution = 1.0
    args.inflation_radius_m = 2.5
    args.laser_max_range_m = 120
    args.num_range = 32
    args.num_bearing = 128
    args.network_file = None

    # Create the map
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)
    return known_map, map_data, pose, goal, args


def run_planning_loop(known_map, map_data, pose, goal, args, unity_path,
                      planner, num_steps=None, do_plan_with_naive=False, do_yield_planner=False):
    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements)
    builder = environments.simulated.WorldBuildingUnityBridge
    robot = lsp.robot.Turtlebot_Robot(pose,
                                      primitive_length=args.step_size,
                                      num_primitives=args.num_primitives,
                                      map_data=map_data)

    with builder(unity_path) as unity_bridge:
        unity_bridge.make_world(world)

        simulator = lsp.simulators.Simulator(known_map,
                                             goal,
                                             args,
                                             unity_bridge=unity_bridge,
                                             world=world)
        simulator.frontier_grouping_inflation_radius = (
            simulator.inflation_radius)

        planning_loop = lsp.planners.PlanningLoop(goal,
                                                  known_map,
                                                  simulator,
                                                  unity_bridge,
                                                  robot,
                                                  args,
                                                  verbose=True)

        for counter, step_data in enumerate(planning_loop):
            # Update the planner objects
            planner.update(
                {'image': step_data['image']},
                step_data['robot_grid'],
                step_data['subgoals'],
                step_data['robot_pose'],
                step_data['visibility_mask'])

            if do_yield_planner:
                yield planner

            if planner is not None and not do_plan_with_naive:
                planning_loop.set_chosen_subgoal(planner.compute_selected_subgoal())

            if num_steps is not None and counter >= num_steps:
                break


def test_lsp_plan_loop_base_no_subgoals(do_debug_plot, unity_path):
    """Confirm that planning with "no subgoals" does not crash."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp.planners.DijkstraPlanner(goal=goal, args=args)
    run_planning_loop(known_map, map_data, pose, goal, args, unity_path,
                      planner, num_steps=20, do_plan_with_naive=True)


def test_lsp_plan_loop_base_dijkstra(do_debug_plot, unity_path):
    """Confirm that planning with the Dijkstra planner does not crash."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp.planners.DijkstraPlanner(goal=goal, args=args)
    run_planning_loop(known_map, map_data, pose, goal, args, unity_path,
                      planner, num_steps=20, do_plan_with_naive=False)


def test_lsp_plan_loop_known_sets_subgoal_nn_inputs(do_debug_plot, unity_path):
    """Confirm that planning with the Dijkstra planner succeeds."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp.planners.KnownSubgoalPlanner(goal=goal, known_map=known_map,
                                               args=args, verbose=True,
                                               do_compute_weightings=True)
    for planner in run_planning_loop(known_map, map_data, pose, goal, args,
                                     unity_path, planner, num_steps=20,
                                     do_plan_with_naive=True,
                                     do_yield_planner=True):

        assert planner.subgoals
        assert all(subgoal.props_set for subgoal in planner.subgoals)
        assert all(hasattr(subgoal, 'nn_input_data') for subgoal in planner.subgoals)

        are_new_subgoals = any([subgoal.just_set for subgoal in planner.subgoals])
        are_old_subgoals = any([not subgoal.just_set for subgoal in planner.subgoals])

        if are_new_subgoals and not are_old_subgoals:
            # If there are only new subgoals, confirm that the images are the same
            total_pixel_values = [subgoal.nn_input_data['image'].sum()
                                  for subgoal in planner.subgoals]
            assert np.std(total_pixel_values) == pytest.approx(0)
        elif are_new_subgoals and are_old_subgoals:
            # If there are both new and old subgoals, confirm that the images are different
            total_pixel_values = [subgoal.nn_input_data['image'].sum()
                                  for subgoal in planner.subgoals]
            assert np.std(total_pixel_values) > 0
            for subgoal in planner.subgoals:
                assert (subgoal.positive_weighting > 1 or subgoal.negative_weighting > 1)
            break

    assert len(planner.subgoals) > 0


def test_lsp_plan_loop_datum_correct_properties(do_debug_plot, unity_path):
    """Show that we can get datum for the new subgoals."""
    known_map, map_data, pose, goal, args = _get_env_and_args()
    planner = lsp.planners.KnownSubgoalPlanner(goal=goal, known_map=known_map,
                                               args=args, verbose=True,
                                               do_compute_weightings=True)
    for planner in run_planning_loop(known_map, map_data, pose, goal, args,
                                     unity_path, planner, num_steps=20,
                                     do_plan_with_naive=True,
                                     do_yield_planner=True):
        subgoal_training_data = planner.get_subgoal_training_data()
        assert all('image' in datum.keys() for datum in subgoal_training_data)
        assert all('goal_loc_x' in datum.keys() for datum in subgoal_training_data)
        assert all('goal_loc_y' in datum.keys() for datum in subgoal_training_data)
        assert all('subgoal_loc_x' in datum.keys() for datum in subgoal_training_data)
        assert all('subgoal_loc_y' in datum.keys() for datum in subgoal_training_data)
        assert all('is_feasible' in datum.keys() for datum in subgoal_training_data)
