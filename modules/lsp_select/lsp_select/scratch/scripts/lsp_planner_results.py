import lsp
import numpy as np
from pathlib import Path


def compute_bandit_cost(env_results, planners, c=4):
    num_pulls_per_bandit = np.zeros(len(planners))
    tot_reward_per_bandit = np.zeros(len(planners))
    all_rewards = []
    num_steps = len(env_results)

    for i in range(num_steps):
        min_idx = np.argmin(num_pulls_per_bandit)
        if num_pulls_per_bandit[min_idx] == 0:
            bandit_ind = min_idx
        else:
            mean_reward_per_bandit = tot_reward_per_bandit / num_pulls_per_bandit
            bandit_ind = np.argmin(mean_reward_per_bandit - c * np.sqrt(np.log(num_pulls_per_bandit.sum())
                                                                        / num_pulls_per_bandit))

        # Data Storage
        reward = env_results[i, bandit_ind]
        tot_reward_per_bandit[bandit_ind] += reward
        num_pulls_per_bandit[bandit_ind] += 1
        all_rewards.append(reward)

    return np.mean(all_rewards), num_pulls_per_bandit


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    args = parser.parse_args()

    planners = ['lspA', 'lspB', 'lspC', 'nonlearned', 'rtmlb']
    envs = ['envA', 'envB', 'envC']

    print('--------------------Planner Results-------------------------')
    results = np.zeros((len(planners), len(envs)))
    for i, planner in enumerate(planners):
        for j, env in enumerate(envs):
            files = Path(args.save_dir).glob(f'cost_{planner}_{env}_*.txt')
            costs = np.array([np.loadtxt(f) for f in sorted(files)])
            results[i, j] = costs.mean()
    print(f'Rows: {planners}\nColumns: {envs}')
    print('Average Cost:')
    print(results)

    print('--------------------RTM-LB Results--------------------')
    for env in envs:
        files = Path(args.save_dir).glob(f'selected_rtmlb_{env}_*.txt')
        selected = np.array([np.loadtxt(f, dtype=str) for f in sorted(files)])
        print(f'--------------------{env}--------------------')
        print('Number of times each planner was selected:')
        for planner, count in zip(*np.unique(selected, return_counts=True)):
            print(f'{planner:<25}{count:>4}')
        idx = np.where(selected == 'DijkstraPlanner[0]')[0]
        print(f'\nLast map where non-learned planner was selected: {idx[-1]}\n')

    print('--------------------UCB-Bandit Results--------------------')
    planners = ['nonlearned', 'lspA', 'lspB', 'lspC']
    bandit_costs = np.zeros(len(envs))
    planner_counts = np.zeros((len(planners), len(envs)))
    for i, env in enumerate(envs):
        env_results = []
        for j, planner in enumerate(planners):
            files = Path(args.save_dir).glob(f'cost_{planner}_{env}_*.txt')
            costs = np.array([np.loadtxt(f) for f in sorted(files)])
            env_results.append(costs)
        env_results = np.array(env_results).T
        avg_cost, counts = compute_bandit_cost(env_results, planners)
        bandit_costs[i] = avg_cost
        planner_counts[:, i] = counts

    print('Average Costs')
    print(envs)
    print(bandit_costs)
    print('Number of times each planner was selected:')
    print(f'Rows: {planners}\nColumns: {envs}')
    print(planner_counts)
