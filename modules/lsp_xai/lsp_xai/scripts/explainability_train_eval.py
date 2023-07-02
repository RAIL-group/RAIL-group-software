import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

import lsp_xai
import lsp
# from lsp import learning
import learning
from lsp_xai.learning.models import ExpNavVisLSP
from common import compute_path_length
from lsp.planners import DijkstraPlanner
from lsp_xai.planners import SubgoalPlanner, KnownSubgoalPlanner
import environments

MODEL_CLASS = ExpNavVisLSP


def get_nn_model_name(model, args, extension):
    """Get the name of the file to which network params will be saved."""
    return os.path.join(args.logdir, f"{model.name}.{extension}.pt")


def save_model(model, args, extension):
    """Save the neural network model weights/params to file.
    Returns the path where the network was saved."""
    net_path = get_nn_model_name(model, args, extension)
    torch.save(model.state_dict(), net_path)
    return net_path


def run_data_gen_eval(args,
                      do_eval,
                      do_explain=False,
                      do_intervene=False,
                      debug_seed=None,
                      logfile=None):
    # Determine what 'seeds' will be looped through
    do_generate_data = (not do_eval)
    do_plan_with_naive = (not do_eval)
    if debug_seed is not None:
        do_generate_data = False
        do_plan_with_naive = False
        args.current_seed = debug_seed

    print(f"Seed: {args.current_seed}")

    args.cirriculum_fraction = 1.0

    if do_explain:
        explanation = run_single_seed_eval(args, do_plan_with_naive,
                                           do_generate_data, do_explain,
                                           do_intervene)
        if explanation is not None:
            env_type = "maze" if 'maze' in args.map_type else "floorplan"
            plot_name = f"explain_{env_type}_{args.current_seed}_{args.explain_at}.png"
            explanation.visualize(os.path.join(args.save_dir, plot_name))
    elif do_intervene:
        did_succeed, dist_learned, dist_baseline, dist_intervene = run_single_seed_eval(
            args, do_plan_with_naive, do_generate_data, do_explain,
            do_intervene)
    else:
        did_succeed, dist_learned, dist_baseline = run_single_seed_eval(
            args, do_plan_with_naive, do_generate_data, do_explain,
            do_intervene)

    if logfile is not None:
        if do_explain:
            with open(logfile, 'a+') as f:
                if explanation is None:
                    f.write(f"[ERR] s: {args.current_seed:4d}")
                else:
                    f.write(f"[SUC] s: {args.current_seed:4d}")
            return

        with open(logfile, 'a+') as f:
            header = ''
            if do_plan_with_naive:
                header = 'Naive'
            else:
                header = 'Learn'
            if debug_seed is not None:
                header += '|dbg'

            err_str = '' if did_succeed else '[ERR]'
            intervene_str = (f" | intervene: {dist_intervene:0.3f}"
                             if do_intervene else '')

            f.write(f"[{header}] {err_str} s: {args.current_seed:4d}"
                    f" | learned: {dist_learned:0.3f}"
                    f"{intervene_str}"
                    f" | baseline: {dist_baseline:0.3f}\n")

    return (lsp.planners.utils.get_csv_file_combined(args),
            lsp.planners.utils.get_csv_file_supervised(args))


def run_single_seed_eval(args,
                         do_plan_with_naive=False,
                         do_write_data=True,
                         do_explain=False,
                         do_intervene=False):
    known_map, map_data, pose, goal = environments.generate.map_and_poses(args)

    if do_plan_with_naive and (do_explain or do_intervene):
        raise ValueError("Can only explain the behavior of the learned agent.")

    # Open the connection to Unity (if desired)
    if args.unity_path is None:
        raise ValueError('Unity Environment Required')

    # Initialize the world and builder objects
    world = environments.simulated.OccupancyGridWorld(
        known_map,
        map_data,
        num_breadcrumb_elements=args.num_breadcrumb_elements,
        min_breadcrumb_signed_distance=4.0 * args.base_resolution,
        min_interlight_distance=3,
        min_light_to_wall_distance=2)
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

        known_planner = KnownSubgoalPlanner(goal=goal,
                                            known_map=known_map,
                                            args=args)
        if not do_write_data and not do_explain and not do_intervene:
            known_planner = None

        if do_explain:
            explanation = lsp_xai.planners.evaluate.run_model_eval(
                args,
                goal,
                known_map,
                simulator,
                unity_bridge,
                get_initialized_robot(),
                eval_planner=SubgoalPlanner(goal=goal, args=args),
                known_planner=known_planner,
                do_write_data=do_write_data,
                do_plan_with_naive=do_plan_with_naive,
                do_explain=do_explain,
                do_plan_with_known=False)
            return explanation

        dat_learned = lsp_xai.planners.evaluate.run_model_eval(
            args,
            goal,
            known_map,
            simulator,
            unity_bridge,
            get_initialized_robot(),
            eval_planner=SubgoalPlanner(goal=goal, args=args),
            known_planner=known_planner,
            do_write_data=do_write_data,
            do_plan_with_naive=do_plan_with_naive,
            do_explain=do_explain,
            do_plan_with_known=False)

        # Run again with an intervention
        if do_intervene:
            # Compute where the intervention should occur.
            dat_intervened = lsp_xai.planners.evaluate.run_model_eval(
                args,
                goal,
                known_map,
                simulator,
                unity_bridge,
                get_initialized_robot(),
                eval_planner=SubgoalPlanner(goal=goal, args=args),
                known_planner=known_planner,
                do_write_data=do_write_data,
                do_plan_with_naive=do_plan_with_naive,
                do_explain=do_explain,
                do_plan_with_known=False,
                intervene_at=dat_learned['start_of_longest_disagreement'])

        if not do_plan_with_naive:
            dat_baseline = lsp_xai.planners.evaluate.run_model_eval(
                args,
                goal,
                known_map,
                simulator,
                unity_bridge,
                get_initialized_robot(),
                eval_planner=DijkstraPlanner(goal=goal, args=args),
                do_write_data=False)
        else:
            dat_baseline = dat_learned

    # Write some plots to file
    if do_intervene:
        num_cols = 3
    else:
        num_cols = 2

    plt.figure(figsize=(16, 16), dpi=150)
    plt.subplot(1, num_cols, 1)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(dat_learned['map'], known_map))
    xs = [p.x for p in dat_learned['path']]
    ys = [p.y for p in dat_learned['path']]
    p = dat_learned['path'][-1]
    plt.plot(ys, xs, 'r')
    plt.plot(p.y, p.x, 'go')
    plt.subplot(1, num_cols, 2)
    plt.imshow(
        lsp.utils.plotting.make_plotting_grid(dat_baseline['map'], known_map))
    xs = [p.x for p in dat_baseline['path']]
    ys = [p.y for p in dat_baseline['path']]
    p = dat_baseline['path'][-1]
    plt.plot(ys, xs, 'r')
    plt.plot(p.y, p.x, 'go')

    if do_intervene:
        plt.subplot(1, num_cols, 3)
        plt.imshow(
            lsp.utils.plotting.make_plotting_grid(dat_intervened['map'],
                                                  known_map))
        xs = [p.x for p in dat_intervened['path']]
        ys = [p.y for p in dat_intervened['path']]
        p = dat_intervened['path'][-1]
        plt.plot(ys, xs, 'r')
        plt.plot(p.y, p.x, 'go')
        if dat_intervened['intervene_pose'] is not None:
            plt.plot(dat_intervened['intervene_pose'][1],
                     dat_intervened['intervene_pose'][0], 'mo')

    if do_write_data:
        img_name = f'data_collect_plots/learned_planner_{args.current_seed}.png'
    elif do_intervene:
        if args.sp_limit_num < 0:
            img_name = f'learned_planner_{args.current_seed}_intervened_allSG.png'
        else:
            img_name = f'learned_planner_{args.current_seed}_intervened_{args.sp_limit_num}SG.png'
    else:
        img_name = f'learned_planner_{args.current_seed}.png'

    plt.savefig(os.path.join(args.save_dir, img_name))
    plt.close()

    if do_intervene:
        return (dat_learned['did_succeed'] and dat_baseline['did_succeed']
                and dat_intervened['did_succeed'],
                compute_path_length(dat_learned['path']),
                compute_path_length(dat_baseline['path']),
                compute_path_length(dat_intervened['path']))
    else:
        return (dat_learned['did_succeed'] and dat_baseline['did_succeed'],
                compute_path_length(dat_learned['path']),
                compute_path_length(dat_baseline['path']))


def train_model(args,
                checkpoint_file,
                all_combined_data_files,
                all_supervised_data_files,
                init=False):
    # Initialize the learning
    torch.manual_seed(args.learning_seed)
    np.random.seed(args.learning_seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"PyTorch Device: {device}")
    model = MODEL_CLASS(args=args)
    tot_index = 0

    # Define the optimizer
    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=0.5)

    # Set up logging
    train_all_writer = SummaryWriter(
        log_dir=os.path.join(args.logdir, "train_all"))

    if init:
        args.network_file = save_model(model, args, 'init')
        print(f"Writing Learning to File: {args.network_file}")
        return None

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model = model.to(device)
    else:
        model = model.to(device)

    # Create the datasets and loaders
    preprocess_function = ExpNavVisLSP.preprocess_data
    train_all_combined_dataset = learning.data.CSVPickleDataset(
        all_combined_data_files)
    train_all_combined_loader = torch.utils.data.DataLoader(
        train_all_combined_dataset, batch_size=1, shuffle=True, num_workers=8)
    train_all_combined_iter = iter(train_all_combined_loader)

    train_all_supervised_dataset = learning.data.CSVPickleDataset(
        all_supervised_data_files, preprocess_function)
    train_all_supervised_loader = torch.utils.data.DataLoader(
        train_all_supervised_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)
    train_all_supervised_iter = iter(train_all_supervised_loader)

    loc_index = 0
    num_decay_steps = 8
    num_data_elements = len(train_all_combined_iter) / num_decay_steps
    max_steps = num_decay_steps * num_data_elements
    epoch_counter = 0

    while loc_index < max_steps:
        # Get the batches
        try:
            batch_all_combined = next(train_all_combined_iter)
        except StopIteration:
            train_all_combined_iter = iter(train_all_combined_loader)
            batch_all_combined = next(train_all_combined_iter)

        try:
            batch_all_supervised = next(train_all_supervised_iter)
        except StopIteration:
            train_all_supervised_iter = iter(train_all_supervised_loader)
            batch_all_supervised = next(train_all_supervised_iter)

        # Update policies in traing data
        print("Updating datum")
        try:
            batch_all_combined = model.update_datum(batch_all_combined, device)
        except ValueError:
            print("Not enough subgoals for comparision update.")
            continue

        if batch_all_combined is None:
            print("Datum is None")
            continue
        print(
            f"  Policy Length: {len(batch_all_combined['target_subgoal_policy']['policy'])}"
        )

        # Compute the 'delta subgoal data' for limiting subgoal num
        delta_subgoal_data = model.get_subgoal_prop_impact(
            batch_all_combined, device, delta_cost_limit=-1e10)

        # Compute loss for 'combined' data
        # (requires re-passing through model due to way
        # subgoal impact is computed)
        # This will occasionally fail when there is only 1 datum
        try:
            out, ind_map = model(batch_all_combined, device)
        except ValueError:
            print("FAILING: combined")
            continue

        loss_all_combined = model.loss(out,
                                       batch_all_combined,
                                       ind_map,
                                       device=device,
                                       writer=train_all_writer,
                                       index=tot_index,
                                       limit_subgoals_num=args.sp_limit_num,
                                       delta_subgoal_data=delta_subgoal_data,
                                       do_include_negative_costs=True,
                                       do_include_limit_costs=False)

        # Compute loss for 'supervised' data
        try:
            out = model.forward_supervised(batch_all_supervised, device)
        except ValueError:
            # This will occasionally happen when there is only 1 datum.
            # Skipping this step avoids the issue.
            continue

        loss_all_supervised = model.loss_supervised(out,
                                                    batch_all_supervised,
                                                    device=device,
                                                    writer=train_all_writer,
                                                    index=tot_index,
                                                    positive_weight=2.0)

        if tot_index % args.summary_frequency == 0:
            model.plot_data_supervised(train_all_writer,
                                       'image',
                                       tot_index,
                                       out=out.detach().cpu(),
                                       data=batch_all_supervised)

        loss = (loss_all_combined + loss_all_supervised.to('cpu'))

        # Perform update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_index += 1
        loc_index += 1

        print(f"Step: {loc_index}/{max_steps} (total: {tot_index})")

        if np.floor(loc_index / num_data_elements) > epoch_counter:
            epoch_counter += 1
            scheduler.step()
            print("Stepping LR")
        else:
            print(
                f"  Epoch: {epoch_counter} | LR: {optimizer.param_groups[0]['lr']}"
            )

    # Save the state to file
    args.network_file = save_model(model, args, 'final')
    return None


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()

    # Add some learning-specific arguments
    parser.add_argument('--training_data_file', nargs='+', type=str)
    parser.add_argument('--logdir',
                        help='Directory in which to store log files',
                        required=True,
                        type=str)
    parser.add_argument('--summary_frequency',
                        default=1000,
                        help='Frequency (in steps) summary is logged to file',
                        type=int)
    parser.add_argument('--num_epochs',
                        default=20,
                        help='Number of epochs to run training',
                        type=int)
    parser.add_argument('--learning_rate',
                        help='Initial learning rate',
                        type=float)
    parser.add_argument('--batch_size',
                        help='Number of data per training iteration batch',
                        type=int)
    parser.add_argument('--relative_positive_weight',
                        default=2.0,
                        help='Positive data relative weight',
                        type=float)

    parser.add_argument('--data_file_base_name', type=str, default='lsp_data')
    parser.add_argument('--debug_seeds', type=int, default=None)
    parser.add_argument('--current_seed', type=int, default=None)
    parser.add_argument('--sp_limit_num', type=int, default=-1)
    parser.add_argument('--learning_seed', type=int, default=8616)
    parser.add_argument('--xent_factor', type=float, default=1.0)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_data_gen', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_explain', action='store_true')
    parser.add_argument('--explain_at', type=int, default=0)
    parser.add_argument('--do_intervene', action='store_true')
    parser.add_argument('--do_init_learning', action='store_true')

    # Parse args
    args = parser.parse_args()
    if args.debug_seeds is not None:
        args.network_file = args.debug_network

    if not (args.do_init_learning or args.do_train):
        # Create the log file
        logfile = os.path.join(args.save_dir, args.logfile_name)
        with open(logfile, "a") as f:
            f.write(f"LOG: {args.current_seed}\n")

    if args.do_init_learning:
        # Save the untrained neural network to file
        checkpoint_file = train_model(args, None, [], [], init=True)
    elif args.do_train:
        args.network_file = get_nn_model_name(MODEL_CLASS, args, 'init')
        all_combined_data_files = glob.glob(
            os.path.join(args.save_dir, "*.combined.csv"))
        all_supervised_data_files = glob.glob(
            os.path.join(args.save_dir, "*.supervised.csv"))
        train_model(args, None, all_combined_data_files,
                    all_supervised_data_files)
    elif args.do_data_gen:
        args.network_file = get_nn_model_name(MODEL_CLASS, args, 'init')
        run_data_gen_eval(args, do_eval=False, logfile=logfile)
    elif args.do_eval:
        args.network_file = get_nn_model_name(MODEL_CLASS, args, 'final')
        run_data_gen_eval(args, do_eval=True, logfile=logfile)
    elif args.do_explain:
        print("Explaining the agent's behavior.")
        args.network_file = get_nn_model_name(MODEL_CLASS, args, 'final')
        run_data_gen_eval(args, do_eval=True, do_explain=True, logfile=logfile)
    elif args.do_intervene:
        args.network_file = get_nn_model_name(MODEL_CLASS, args, 'final')
        run_data_gen_eval(args,
                          do_eval=True,
                          do_intervene=True,
                          logfile=logfile)
    else:
        print("No operation selected. Ending without computation.")
