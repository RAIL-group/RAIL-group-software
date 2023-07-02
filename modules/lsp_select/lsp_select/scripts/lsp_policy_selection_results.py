import lsp
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

NUM_TRIALS = 100
NUM_SAMPLING = 200


def compute_ucb_bandit_cost(env, planners, c=100, random_seed=42):
    """Computes UCB bandit cost when deployed in a given environment.
    Cost of a planner for each trial is read from file based on
    chosen planner, all available planers, environment name and map seed.

    Returns cumulative costs over trials, selection rates and chosen planner indices.
    """
    seeds = np.arange(*env_seeds[env])
    np.random.seed(random_seed)

    # Shuffle seeds to randomize the order of maps
    np.random.shuffle(seeds)
    tot_cost_per_planner = np.zeros(len(planners))
    num_selection_per_planner = np.zeros(len(planners))

    chosen_indices = []
    all_costs = []
    for _ in range(NUM_TRIALS):
        seed = np.random.choice(seeds)
        num_trials = num_selection_per_planner.sum()
        if num_trials < 1:
            num_trials = 1
        mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner + 0.001)

        # Compute the planner index with minimum UCB cost
        ucb_cost = mean_cost_per_planner - c * np.sqrt(np.log(num_trials)
                                                       / ((num_selection_per_planner + 0.001)))
        min_value = np.min(ucb_cost)
        min_indices = np.where(ucb_cost == min_value)[0]
        min_idx = np.random.choice(min_indices)
        # Update the chosen planner
        chosen_planner = planners[min_idx]
        chosen_indices.append(min_idx)

        # Get cost of chosen planner for current trial
        cost_file = Path(args.save_dir) / f'cost_{chosen_planner}_all_{all_planners}_{env}_{seed}.txt'
        cost = np.loadtxt(cost_file)[min_idx]

        # Update costs and times selected
        all_costs.append(cost)
        tot_cost_per_planner[min_idx] += cost
        num_selection_per_planner[min_idx] += 1

    return (np.cumsum(all_costs) / (np.arange(NUM_TRIALS) + 1),
            num_selection_per_planner / NUM_TRIALS, chosen_indices)


def compute_base_planner_costs(env, planners, seed=42):
    """Computes cumulative costs of each planner over trials in an environment
    without performing any selection.
    """
    seeds = np.arange(*env_seeds[env])
    np.random.seed(seed)
    np.random.shuffle(seeds)
    costs_per_planner = np.zeros((NUM_TRIALS, len(planners)))
    for i in range(NUM_TRIALS):
        seed = np.random.choice(seeds)
        for j, chosen_planner in enumerate(planners):
            cost_file = Path(args.save_dir) / f'cost_{chosen_planner}_all_{all_planners}_{env}_{seed}.txt'
            cost = np.loadtxt(cost_file)
            costs_per_planner[i, j] = cost[j]

    return np.cumsum(costs_per_planner, axis=0) / (np.arange(NUM_TRIALS).reshape(-1, 1) + 1)


def compute_lbcost_wavg(env, chosen_planner, env_seed, prob_shortcut=0):
    """Computes weighted average of optimistic and simply-connected lower bound cost
    for a planner in a given map seed based on likelihood of finding a shorter path to goal
    in the environment. For chosen planner, true cost is computed.
    """
    lb_costs_file = Path(args.save_dir) / f'lbc_{chosen_planner}_all_{all_planners}_{env}_{env_seed}.txt'
    cost_file = Path(args.save_dir) / f'cost_{chosen_planner}_all_{all_planners}_{env}_{env_seed}.txt'
    true_costs = np.loadtxt(cost_file)
    lb_costs = np.loadtxt(lb_costs_file)
    optimistic_lb = lb_costs[:, 0]
    simply_connected_lb = lb_costs[:, 1]
    # Use simply connected lb values if optimistic lb values are infinity
    optimistic_lb[np.isinf(optimistic_lb)] = simply_connected_lb[np.isinf(optimistic_lb)]
    # Compute weighted average
    wavg = prob_shortcut * optimistic_lb + (1 - prob_shortcut) * simply_connected_lb
    chosen_planner_idx = planners.index(chosen_planner)
    # For chosen planner, true cost is returned instead
    wavg[chosen_planner_idx] = true_costs[chosen_planner_idx]

    return wavg


def compute_lb_selection_cost(env, planners, c=100, prob_shortcut=0, random_seed=42):
    """Computes Const-UCB (ours) cost when deployed in a given environment.
    The function 'compute_lbcost_wavg' is used to get true cost for chosen planner and
    weighted lowerbound costs for other planners which are used to select among planners
    using modified UCB bandit appoach.

    Returns cumulative costs over trials, selection rates and chosen planner indices.
    """
    seeds = np.arange(*env_seeds[env])
    np.random.seed(random_seed)
    np.random.shuffle(seeds)
    # Store true cost (first row) and simulated lb costs (second row)
    tot_cost_per_planner = np.zeros((2, len(planners)))
    num_selection_per_planner = np.zeros((2, len(planners)))

    chosen_indices = []
    all_costs = []
    for _ in range(NUM_TRIALS):
        seed = np.random.choice(seeds)
        num_trials = num_selection_per_planner[0].sum()
        if num_trials < 1:
            num_trials = 1
        # Compute mean costs for each planner
        mean_cost_per_planner = tot_cost_per_planner / (num_selection_per_planner + 0.001)
        # Compute weighted average cost based on true and simulated lb costs
        cost_wavg = (num_selection_per_planner[0] * mean_cost_per_planner[0] +
                     num_selection_per_planner[1] * mean_cost_per_planner[1]) / num_trials
        # Compute exploration magnitude of UCB
        bandit_exploration_magnitude = np.sqrt(np.log(num_trials) /
                                               (num_selection_per_planner[0] + 0.001))
        # Compute UCB bandit cost as usual
        bandit_cost = (mean_cost_per_planner[0] - c * bandit_exploration_magnitude)
        # Compute final cost used for selection (Const-UCB cost)
        our_cost = np.maximum(bandit_cost, cost_wavg)
        # Compute the planner index with minimum Const-UCB cost
        min_idx = np.argmin(our_cost)
        # Update the chosen planner
        chosen_planner = planners[min_idx]
        chosen_indices.append(min_idx)

        # Get true cost (for chosen planner) and simulated lb costs (for other planners) for current trial
        costs = compute_lbcost_wavg(env, chosen_planner, seed, prob_shortcut=prob_shortcut)

        # Make updates to planner costs and times selected
        all_costs.append(costs[min_idx])
        costs_try = np.zeros_like(costs)
        costs_try[min_idx] = costs[min_idx]
        costs_sim = costs.copy()
        costs_sim[min_idx] = 0
        tot_cost_per_planner[0] += costs_try
        tot_cost_per_planner[1] += costs_sim
        num_selection_per_planner[0, min_idx] += 1
        num_selection_per_planner[1] += 1
        num_selection_per_planner[1, min_idx] -= 1

    return (np.cumsum(all_costs) / (np.arange(NUM_TRIALS) + 1),
            num_selection_per_planner[0] / NUM_TRIALS, chosen_indices)


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--experiment_type', choices=['maze', 'office'])
    parser.add_argument('--start_seeds', type=int, nargs='+', default=[2000, 3000, 4000])
    parser.add_argument('--num_seeds', type=int, default=150)
    args = parser.parse_args()

    if args.experiment_type == 'maze':
        planners = ['nonlearned', 'lspA', 'lspB', 'lspC']
        planner_names = ['Non-learned', 'LSP-Maze-Green', 'LSP-Maze-Gray', 'LSP-Maze-Random']
        planner_colors = ['brown', 'green', 'gray', 'darkorange']
        envs = ['envA', 'envB', 'envC']
        env_names = ['Maze-Green', 'Maze-Gray', 'Maze-Random']
        env_seeds = {'envA': (args.start_seeds[0], args.start_seeds[0] + args.num_seeds),
                     'envB': (args.start_seeds[1], args.start_seeds[1] + args.num_seeds),
                     'envC': (args.start_seeds[2], args.start_seeds[2] + args.num_seeds)}

    elif args.experiment_type == 'office':
        planners = ['nonlearned', 'lspmaze', 'lspoffice', 'lspofficewallswap']
        planner_names = ['Non-learned', 'LSP-Maze-Green', 'LSP-Office-Base', 'LSP-Office-Diff']
        planner_colors = ['brown', 'green', 'cyan', 'purple']
        envs = ['mazeA', 'office', 'officewall']
        env_names = ['Maze-Green', 'Office-Base', 'Office-Diff']
        env_seeds = {'mazeA': (args.start_seeds[0], args.start_seeds[0] + args.num_seeds),
                     'office': (args.start_seeds[1], args.start_seeds[1] + args.num_seeds),
                     'officewall': (args.start_seeds[2], args.start_seeds[2] + args.num_seeds)}

    else:
        raise ValueError(f'Experiment name "{args.experiment_type}" is not valid.')

    all_planners = '_'.join(planners)

    trials_to_print = np.array([10, 40, 100]) - 1
    trial_markers = ['^', 'd', 's']
    trial_marker_size = 9
    ucb_color = 'deepskyblue'
    best_policy_color = 'darkorange'
    fill_alpha = 0.08
    xticks = list(range(0, NUM_TRIALS + 1, 20))
    xticks[0] = 1

    env_planner_costs = {}

    print('------------------------Base Planner Results----------------------------')
    for i, env in enumerate(envs):
        print(f'-------------------------------{env_names[i]}-------------------------------')
        dat = [compute_base_planner_costs(env, planners, seed=seed) for seed in range(NUM_SAMPLING)]
        dat = np.array(dat)
        planner_costs = {}
        for j, planner in enumerate(planners):
            all_runs = dat[:, :, j]
            planner_avg_cost = np.mean(all_runs, axis=0)[-1]
            planner_costs[j] = planner_avg_cost
            print(f'Incurred Cost [{planner_names[j]:<20}]: {planner_avg_cost:.2f}')
        env_planner_costs[i] = planner_costs

    if args.experiment_type == 'office':
        probs = [1.0, 0.5, 0.0]
        tags = [r'$C^{lb,opt}$', r'$C_{p=0.5}^{lb,wgt}$', r'$C^{lb,s.c.}$']
        y_labels = ['Optimistic LB', 'Weighted LB', 'Simply-Connected LB']
        colors = ['green', 'blue', 'purple']
    elif args.experiment_type == 'maze':
        probs = [1.0, 0.0]
        tags = [r'$C^{lb,opt}$', r'$C^{lb,s.c.}$']
        y_labels = ['Optimistic LB', 'Simply-Connected LB']
        colors = ['green', 'purple']

    fig_costs, axs_costs = plt.subplots(1, len(envs), figsize=(17, 3))
    fig_regret, axs_regret = plt.subplots(1, len(envs), figsize=(17, 2.1))
    fig_costs.subplots_adjust(wspace=0.3, top=0.95, bottom=0.03, left=0.06, right=0.985)
    fig_regret.subplots_adjust(wspace=0.3, top=1, bottom=0.276, left=0.06, right=0.985)

    for i, env in enumerate(envs):
        print(f'\n-------------------------------{env_names[i]}-------------------------------')
        print('----------------------UCB-Bandit Results-----------------------')
        dat = [compute_ucb_bandit_cost(env, planners, random_seed=seed) for seed in range(NUM_SAMPLING)]
        all_runs, pull_rates, all_chosen_indx = zip(*dat)
        avg_costs_ucb = np.mean(all_runs, axis=0)
        p10_costs_ucb = np.percentile(all_runs, 10, axis=0)
        p90_costs_ucb = np.percentile(all_runs, 90, axis=0)

        best_asymp_cost = min(env_planner_costs[i].values())

        # UCB Cost
        plt.sca(axs_costs[i])
        axs_costs[i].spines['top'].set_visible(False)
        axs_costs[i].spines['right'].set_visible(False)

        plt.plot(range(1, NUM_TRIALS + 1),
                 avg_costs_ucb,
                 color=ucb_color,
                 label='UCB-Bandit (baseline)')
        plt.fill_between(range(1, NUM_TRIALS + 1),
                         p10_costs_ucb,
                         p90_costs_ucb,
                         alpha=fill_alpha,
                         color=ucb_color)

        for m, trial in enumerate(trials_to_print):
            plt.plot(trial + 1,
                     avg_costs_ucb[trial],
                     marker=trial_markers[m],
                     markersize=trial_marker_size,
                     color=ucb_color)
            print(f'Trial {trial + 1} Cost: {avg_costs_ucb[trial]:.2f}')

        plt.xticks(xticks, fontsize='x-large')
        plt.xlim([1, NUM_TRIALS + 2])
        plt.gca().set_xticklabels([])
        plt.ylabel('Avg. Navigation Cost', fontsize='x-large')
        if 'maze' in env_names[i].lower():
            plt.ylim([0, 400])
            plt.yticks(range(0, 401, 100), fontsize='x-large')
        else:
            plt.ylim([0, 1300])
            plt.yticks(range(0, 1300, 300), fontsize='x-large')

        regrets_ucb = np.cumsum(all_runs - best_asymp_cost, axis=1)
        avg_regrets_ucb = regrets_ucb.mean(0)

        # Const-UCB Pull Rate
        print('Percentage of times each planner was selected:')
        selection_counts = [[] for _ in planners]
        for chosen_indices in all_chosen_indx:
            for planner_idx, count in zip(*np.unique(chosen_indices, return_counts=True)):
                selection_counts[planner_idx].append(count)
        mean_counts = np.array([np.mean(counts) for counts in selection_counts])
        print(planner_names)
        print(mean_counts / NUM_TRIALS * 100)

        # UCB Regret
        plt.sca(axs_regret[i])
        axs_regret[i].spines['top'].set_visible(False)
        axs_regret[i].spines['right'].set_visible(False)
        plt.plot(range(1, NUM_TRIALS + 1),
                 avg_regrets_ucb,
                 color=ucb_color,
                 label='UCB-Bandit (baseline)')
        plt.fill_between(range(1, NUM_TRIALS + 1),
                         np.percentile(regrets_ucb, 10, axis=0),
                         np.percentile(regrets_ucb, 90, axis=0),
                         alpha=fill_alpha,
                         color=ucb_color)

        for m, trial in enumerate(trials_to_print):
            plt.plot(trial + 1,
                     avg_regrets_ucb[trial],
                     marker=trial_markers[m],
                     markersize=trial_marker_size,
                     color=ucb_color,
                     fillstyle='none')
            print(f'Trial {trial + 1} Regret: {avg_regrets_ucb[trial]:.1f}')

        plt.xlabel(f'Num of Trials ({r"$k$"}) in {env_names[i]}', fontsize='x-large')
        plt.xticks(xticks, fontsize='x-large')
        plt.xlim([1, NUM_TRIALS + 2])
        plt.ylabel('Cumulative Regret', fontsize='x-large')
        if 'maze' in env_names[i].lower():
            plt.ylim([-3000, 9000])
            plt.yticks(range(-3000, 9000, 3000), fontsize='x-large')
        else:
            plt.ylim([-4000, 15000])
            plt.yticks(range(-4000, 15000, 4000), fontsize='x-large')

        print('----------------------Constrained-UCB Results--------------------------')
        for k, p_short in enumerate(probs):
            print(f'--------------------------{p_short=}--------------------------')
            dat = [compute_lb_selection_cost(env, planners, prob_shortcut=p_short, random_seed=seed)
                   for seed in range(NUM_SAMPLING)]
            all_runs, pull_rates, all_chosen_indx = zip(*dat)
            avg_costs_const_ucb = np.mean(all_runs, axis=0)
            p10_costs_const_ucb = np.percentile(all_runs, 10, axis=0)
            p90_costs_const_ucb = np.percentile(all_runs, 90, axis=0)

            # Const-UCB Cost
            plt.sca(axs_costs[i])
            axs_costs[i].spines['top'].set_visible(False)
            axs_costs[i].spines['right'].set_visible(False)

            plt.plot(range(1, NUM_TRIALS + 1),
                     avg_costs_const_ucb,
                     color=colors[k],
                     label=f'Const-UCB: {tags[k]} (ours)')
            plt.fill_between(range(1, NUM_TRIALS + 1),
                             p10_costs_const_ucb,
                             p90_costs_const_ucb,
                             alpha=fill_alpha,
                             color=colors[k])

            plt.sca(axs_costs[i])
            for m, trial in enumerate(trials_to_print):
                plt.plot(trial + 1, avg_costs_const_ucb[trial],
                         marker=trial_markers[m],
                         markersize=trial_marker_size,
                         color=colors[k])
                print(f'Trial {trial + 1} Cost: {avg_costs_const_ucb[trial]:.2f}')

            regrets_const_ucb = np.cumsum(all_runs - best_asymp_cost, axis=1)
            avg_regrets_const_ucb = regrets_const_ucb.mean(0)

            # Const-UCB Pull Rate
            print('Number of times each planner was selected:')
            selection_counts = [[] for _ in planners]
            for chosen_indices in all_chosen_indx:
                for planner_idx, count in zip(*np.unique(chosen_indices, return_counts=True)):
                    selection_counts[planner_idx].append(count)
            mean_counts = [np.mean(counts) for counts in selection_counts]
            print(planner_names)
            print(mean_counts)

            # Const-UCB Regret
            plt.sca(axs_regret[i])
            plt.plot(range(1, NUM_TRIALS + 1),
                     avg_regrets_const_ucb,
                     color=colors[k],
                     label=f'Const-UCB: {tags[k]}')
            plt.fill_between(range(1, NUM_TRIALS + 1),
                             np.percentile(regrets_const_ucb, 10, axis=0),
                             np.percentile(regrets_const_ucb, 90, axis=0),
                             alpha=fill_alpha,
                             color=colors[k])

            for m, trial in enumerate(trials_to_print):
                plt.plot(trial + 1,
                         avg_regrets_const_ucb[trial],
                         marker=trial_markers[m],
                         markersize=trial_marker_size,
                         color=colors[k],
                         fillstyle='none')
                print(f'Trial {trial + 1} Regret: {avg_regrets_const_ucb[trial]:.1f}')

        plt.sca(axs_regret[i])
        plt.hlines(y=0, xmin=1, xmax=NUM_TRIALS,
                   colors=best_policy_color, linestyles='--', label='Best Single Policy')
        plt.sca(axs_costs[i])
        plt.hlines(y=best_asymp_cost, xmin=1, xmax=NUM_TRIALS,
                   colors=best_policy_color, linestyles='--', label='Best Single Policy')
        plt.legend(fontsize='large')
    fig_costs.savefig(Path(args.save_dir) / f'results_{args.experiment_type}_costs.png')
    fig_regret.savefig(Path(args.save_dir) / f'results_{args.experiment_type}_regret.png')
    print(f'Results plots saved in {args.save_dir} as '
          f'results_{args.experiment_type}_costs.png'
          f'and results_{args.experiment_type}_regret.png')
    plt.show()
