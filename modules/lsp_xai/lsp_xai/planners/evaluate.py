import math
import numpy as np
import time
import logging
import gzip
import _pickle as cPickle
import os
import torch

import gridmap
from gridmap.constants import UNOBSERVED_VAL
import lsp
from lsp.planners.utils import COMPRESS_LEVEL
import lsp_xai

logging.basicConfig(filename="/data/interp.log", level=logging.DEBUG)


def run_model_eval(args,
                   goal,
                   known_map,
                   simulator,
                   unity_bridge,
                   robot,
                   eval_planner,
                   known_planner=None,
                   do_write_data=False,
                   do_plan_with_naive=False,
                   do_plan_with_known=False,
                   return_planner_after_steps=None,
                   do_explain=False,
                   intervene_at=None):
    """Main function for evaluation (and optional data writing) for a given model class."""
    if do_write_data and known_planner is None:
        raise ValueError(
            "Writing data requires the known_planner not be 'None'.")

    counter = 0
    count_since_last_turnaround = 100
    fn_start_time = time.time()
    travel_data = []
    robot_grid = UNOBSERVED_VAL * np.ones(known_map.shape)

    did_succeed = True
    agrees_with_oracle = []
    intervene_pose = None
    did_intervene = False
    chosen_planner = 'chosen'
    planners = {'chosen': eval_planner}
    if known_planner is not None:
        planners['known'] = known_planner
    elif do_plan_with_known:
        raise ValueError("Cannot plan with known planner if no known planner "
                         "provided. 'known_planner' cannot be None or"
                         "'do_plan_with_known' should be False.")

    # Main planning loop
    while (math.fabs(robot.pose.x - goal.x) >= 3 * args.step_size
           or math.fabs(robot.pose.y - goal.y) >= 3 * args.step_size):
        logger = logging.getLogger("evaluate")

        if not args.silence:
            print(("Goal: {}, {}".format(goal.x, goal.y)))
            print(("Robot: {}, {} [motion: {}]".format(robot.pose.x,
                                                       robot.pose.y,
                                                       robot.net_motion)))
            print(f"Counter: {counter} | Count since last turnaround: "
                  f"{count_since_last_turnaround}")

        stime = time.time()

        # Compute observations and update map
        pano_image = simulator.get_image(robot)
        _, robot_grid, visible_region = (
            simulator.get_laser_scan_and_update_map(robot, robot_grid, True))
        logger.debug(f"[{counter}] time to observation: {time.time() - stime}")

        # Compute intermediate map grids for planning
        update_map_time = time.time()
        visibility_mask = gridmap.utils.inflate_grid(visible_region, 1.8, -0.1,
                                                     1.0)
        inflated_grid = simulator.get_inflated_grid(robot_grid, robot)
        inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            inflated_grid, robot.pose)
        logger.debug(
            f"[{counter}] time to update map: {time.time() - update_map_time}")

        # Compute the subgoal
        subgoal_time = time.time()
        subgoals = simulator.get_updated_frontier_set(inflated_grid, robot,
                                                      set())
        logger.debug(
            f"[{counter}] time to compute subgoals (# {len(subgoals)}): {time.time() - subgoal_time}"
        )

        # Update the planner objects
        planner_update_time = time.time()
        for planner in planners.values():
            planner.update({'image': pano_image}, robot_grid, subgoals,
                           robot.pose, visibility_mask)
        logger.debug(
            f"[{counter}] time to update planners: {time.time() - planner_update_time}"
        )

        # Compute the subgoals for the different planners
        chosen_subgoal_time = time.time()
        chosen_subgoal = planners[chosen_planner].compute_selected_subgoal()
        logger.debug(
            f"[{counter}] time to compute_selected_subgoal: {time.time() - chosen_subgoal_time}"
        )

        if known_planner is not None and chosen_subgoal is not None:
            target_subgoal_time = time.time()
            target_subgoal = planners['known'].compute_selected_subgoal()
            logger.debug(
                f"[{counter}] time to target_selected_subgoal: {time.time() - target_subgoal_time}"
            )

            # Say whether the two planners agree
            agrees_with_oracle.append(chosen_subgoal == target_subgoal)

            def get_subgoal_path_angle(subgoal):
                planning_grid = lsp.core.mask_grid_with_frontiers(
                    inflated_grid, subgoals, do_not_mask=subgoal)

                # Check that the plan is feasible and compute path
                _, get_path = gridmap.planning.compute_cost_grid_from_position(
                    planning_grid, [goal.x, goal.y], use_soft_cost=True)
                _, path = get_path([robot.pose.x, robot.pose.y],
                                   do_sparsify=True,
                                   do_flip=True)
                try:
                    return np.arctan2(path[1][1] - path[1][0],
                                      path[0][1] - path[0][0])
                except IndexError:
                    return 0.0

            syaw_ch = get_subgoal_path_angle(chosen_subgoal)
            syaw_ta = get_subgoal_path_angle(target_subgoal)

            rel_angle = abs(((syaw_ch - syaw_ta) + np.pi) % (2 * np.pi) -
                            np.pi)
        elif do_explain:
            raise ValueError("known_planner is None yet is required "
                             "to generate explainations.")
        else:
            target_subgoal = None

        if (do_explain and counter >= args.explain_at):
            try:
                explanation = planners[
                    chosen_planner].generate_counterfactual_explanation(
                        target_subgoal,
                        limit_num=args.sp_limit_num,
                        do_freeze_selected=False)
                return explanation
            except:  # noqa
                return None

        if (intervene_at is not None and not did_intervene
                and counter >= intervene_at and rel_angle > np.pi / 2):
            # Intervene: change the learned model to 'fix' behavior
            try:
                update_counter = planners[chosen_planner].update_counter
                planners[chosen_planner].update_counter = 'intervention'
                lsp.planners.utils.write_comparison_training_datum(
                    planners[chosen_planner], target_subgoal)
                planners[chosen_planner].update_counter = update_counter

                base_name = f"learned_planner_{args.current_seed}"
                net_name_base = f"{base_name}_intervention_weights"

                # Save the base model weights
                torch.save(planners[chosen_planner].model.state_dict(),
                           os.path.join(args.save_dir, net_name_base + '.before.pt'))

                # Compute the explanation
                explanation = planners[chosen_planner].generate_counterfactual_explanation(
                    target_subgoal,
                    limit_num=args.sp_limit_num,
                    do_freeze_selected=False,
                    keep_changes=True,
                    margin=0.1)

                # Save the explanation image to file
                plot_name = f"{base_name}_intervention_exp_{intervene_at}.png"
                explanation.visualize(os.path.join(args.save_dir, plot_name))

                chosen_subgoal = planners[
                    chosen_planner].compute_selected_subgoal()
                intervene_pose = [robot.pose.x, robot.pose.y]
                did_intervene = True
                if (not chosen_subgoal == target_subgoal):
                    planners[
                        chosen_planner].generate_counterfactual_explanation(
                            target_subgoal,
                            limit_num=args.sp_limit_num,
                            do_freeze_selected=False,
                            keep_changes=True,
                            margin=5.0)
                    chosen_subgoal = planners[
                        chosen_planner].compute_selected_subgoal()
                if (not chosen_subgoal == target_subgoal):
                    planners[
                        chosen_planner].generate_counterfactual_explanation(
                            target_subgoal,
                            limit_num=args.sp_limit_num,
                            do_freeze_selected=False,
                            keep_changes=True,
                            margin=10.0)
                    chosen_subgoal = planners[
                        chosen_planner].compute_selected_subgoal()

                # Save the base model weights
                torch.save(planners[chosen_planner].model.state_dict(),
                           os.path.join(args.save_dir, net_name_base + '.after.pt'))

                assert chosen_subgoal == target_subgoal
            except TypeError:
                pass
            except:  # noqa
                did_succeed = False
                raise ValueError
                break

        stime = time.time()

        if do_write_data:
            # Generate and write the supervised data
            lsp.planners.utils.write_supervised_training_datum_oriented(
                planners['known'])

            # Comparison Data
            did_write, pickle_name = (
                lsp.planners.utils.write_comparison_training_datum(
                    planners[chosen_planner], target_subgoal))

            if did_write:
                _, _, known_distance = planners['known'].compute_path_to_goal()
                travel_data.append({
                    'pickle_name': pickle_name,
                    'robot_motion': robot.net_motion,
                    'known_distance': known_distance,
                })
                print(f"Time to write data: {time.time() - stime}")

        # Plotting
        if lsp_xai.DEBUG_PLOT_PLAN and counter % 2 == 0:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 6))
            planners[chosen_planner].plot_map_with_plan(None, robot.all_poses, image=pano_image)
            if intervene_at is not None:
                plt.savefig(f"/data/tmp/planner_intervene_{counter:04}.png", dpi=300)
            else:
                plt.savefig(f"/data/tmp/planner_base_{counter:04}.png", dpi=300)

            plt.clf()
            plt.close(fig)
            plt.close('all')
            import gc
            gc.collect()

        # Mask grid with chosen subgoal (if not None)
        # and compute the cost grid for motion planning.
        if do_plan_with_known and target_subgoal is not None:
            # Plan using the known planner
            print("Planning with known planner.")
            planning_grid = lsp.core.mask_grid_with_frontiers(
                inflated_grid, subgoals, do_not_mask=target_subgoal)
        elif do_plan_with_naive or chosen_subgoal is None:
            # Plan using the 'naive' planner
            print("Planning with naive planner.")
            planning_grid = lsp.core.mask_grid_with_frontiers(
                inflated_grid,
                [],
            )
        else:
            # Plan using the learned planner
            print("Planning with learned planner.")
            planning_grid = lsp.core.mask_grid_with_frontiers(
                inflated_grid, subgoals, do_not_mask=chosen_subgoal)

        # Check that the plan is feasible and compute path
        cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
            planning_grid, [goal.x, goal.y], use_soft_cost=True)
        did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                  do_sparsify=True,
                                  do_flip=True)

        # Move the robot
        motion_primitives = robot.get_motion_primitives()
        do_use_path = (count_since_last_turnaround > 10)
        costs, _ = lsp.primitive.get_motion_primitive_costs(
            planning_grid,
            cost_grid,
            robot.pose,
            path,
            motion_primitives,
            do_use_path=do_use_path)
        if abs(min(costs)) < 1e10:
            primitive_ind = np.argmin(costs)
            robot.move(motion_primitives, primitive_ind)
            if primitive_ind == len(motion_primitives) - 1:
                count_since_last_turnaround = -1
        else:
            # Force the robot to return to known space
            cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
                planning_grid, [goal.x, goal.y],
                use_soft_cost=True,
                obstacle_cost=1e5)
            did_plan, path = get_path([robot.pose.x, robot.pose.y],
                                      do_sparsify=True,
                                      do_flip=True)
            costs, _ = lsp.primitive.get_motion_primitive_costs(
                planning_grid,
                cost_grid,
                robot.pose,
                path,
                motion_primitives,
                do_use_path=False)
            robot.move(motion_primitives, np.argmin(costs))

        # Check that the robot is not 'stuck'.
        if robot.max_travel_distance(
                num_recent_poses=100) < 5 * args.step_size:
            print("Planner stuck")
            did_succeed = False
            break

        if robot.net_motion > 4000:
            print("Reached maximum distance.")
            did_succeed = False
            break

        logger.debug(f"[{counter}] total step time: {time.time() - stime}")
        counter += 1
        count_since_last_turnaround += 1
        if not args.silence:
            print("")

        # For testing and debugging purposes
        if return_planner_after_steps is not None and counter > return_planner_after_steps:
            return eval_planner

    if not args.silence:
        print("TOTAL TIME:", time.time() - fn_start_time)

    # Update the pickles with the final motion data
    if do_write_data:
        net_travel = robot.net_motion
        planners['known'].robot_pose = robot.pose
        did_plan, _, final_known_distance = planners[
            'known'].compute_path_to_goal()

        if not did_succeed:
            final_known_distance += 1000
        if not did_plan:
            raise ValueError(
                f"Robot ending away from goal (seed: {args.current_seed}.")
        for dat in travel_data:
            # Open the pickle
            pickle_name = dat['pickle_name']
            with gzip.GzipFile(pickle_name, 'rb') as pfile:
                pdat = cPickle.load(pfile)

            # Update the data
            pdat['net_cost_remaining'] = (final_known_distance + net_travel -
                                          dat['robot_motion'])
            pdat['net_cost_remaining_known'] = dat['known_distance']

            # Dump the pickle
            with gzip.GzipFile(pickle_name, 'wb',
                               compresslevel=COMPRESS_LEVEL) as f:
                cPickle.dump(pdat, f, protocol=-1)

    print("AGREES")
    print(agrees_with_oracle)

    agrees_with_oracle
    idx_pairs = np.where(
        np.diff(
            np.hstack(([False], np.logical_not(agrees_with_oracle),
                       [False]))))[0].reshape(-1, 2)
    try:
        start_longest_disagreement = (
            idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 0])
    except ValueError:
        start_longest_disagreement = 0

    return {
        'did_succeed': did_succeed,
        'map': robot_grid,
        'path': robot.all_poses,
        'agrees_with_oracle': agrees_with_oracle,
        'start_of_longest_disagreement': start_longest_disagreement,
        'intervene_pose': intervene_pose
    }
