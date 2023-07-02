import lsp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def get_cost_learned(cost_data, args):
    do_corrupt = np.random.binomial(1, args.corrupt_pose_prob, size=args.no_of_maps)
    cost = (cost_data['learned_corrupted'][:, 1] * do_corrupt
            + cost_data['learned_uncorrupted'][:, 1] * (1 - do_corrupt))
    return np.mean(cost)


def get_cost_non_learned(cost_data, args):
    do_corrupt = np.random.binomial(1, args.corrupt_pose_prob, size=args.no_of_maps)
    cost = (cost_data['non_learned_corrupted'][:, 1] * do_corrupt
            + cost_data['non_learned_uncorrupted'][:, 1] * (1 - do_corrupt))
    return np.mean(cost)


def get_cost_ideal_model_selection(cost_data, args):
    do_corrupt = np.random.binomial(1, args.corrupt_pose_prob, size=args.no_of_maps)
    cost_learned = (cost_data['learned_corrupted'][:, 1] * do_corrupt
                    + cost_data['learned_uncorrupted'][:, 1] * (1 - do_corrupt))
    cost_non_learned = (cost_data['non_learned_corrupted'][:, 1] * do_corrupt
                        + cost_data['non_learned_uncorrupted'][:, 1] * (1 - do_corrupt))
    switch = do_corrupt
    cost = cost_learned * (1 - switch) + cost_non_learned * switch
    return np.mean(cost)


def get_cost_our_model_selection(cost_data, args):
    do_corrupt = np.random.binomial(1, args.corrupt_pose_prob, size=args.no_of_maps)
    nll_learned = (cost_data['learned_corrupted'][:, 2] * do_corrupt
                   + cost_data['learned_uncorrupted'][:, 2] * (1 - do_corrupt))
    num_samples = (cost_data['learned_corrupted'][:, 3] * do_corrupt
                   + cost_data['learned_uncorrupted'][:, 3] * (1 - do_corrupt))
    if args.stepwise_eval:
        avg_nll = ((np.cumsum(nll_learned) + args.prior_nll)
                   / (np.cumsum(num_samples) + args.prior_num_samples))
        avg_nll = -np.log(avg_nll)
    else:
        avg_nll = ((np.cumsum(-np.log(nll_learned / num_samples)) + args.prior_nll)
                   / (np.arange(1, args.no_of_maps + 1) + args.prior_num_samples))
    switch = avg_nll > args.threshold
    switch_idx = np.where(switch)[0]
    switch_idx = switch_idx[0] if len(switch_idx) else len(switch) - 1
    args.switch_idx = switch_idx
    switch[switch_idx:] = True
    cost_learned = (cost_data['learned_corrupted'][:, 1] * do_corrupt
                    + cost_data['learned_uncorrupted'][:, 1] * (1 - do_corrupt))
    cost_non_learned = (cost_data['non_learned_corrupted'][:, 1] * do_corrupt
                        + cost_data['non_learned_uncorrupted'][:, 1] * (1 - do_corrupt))
    cost = cost_learned * (1 - switch) + cost_non_learned * switch
    return np.mean(cost)


def read_data(data_files, args, aggregate=True):
    path = Path(args.save_dir) / 'data'
    files = {x: path.glob(f'{x}_*.gz') for x in data_files}
    cost_data = {x: [] for x in data_files}
    for model, data_path in files.items():
        for f in sorted(data_path):
            seed = re.findall(r'\d+', f.name)
            seed = int(seed[0]) if len(seed) != 0 else -1
            data = np.loadtxt(f)
            data = data.reshape(-1, 2) if data.size != 1 else data
            cost_data[model].append([seed, data])
    if not aggregate:
        return cost_data
    # Aggregate data per map
    for model, seed_data in cost_data.items():
        map_data = []
        for seed, d in seed_data:
            if d.ndim == 0:
                map_data.append([seed, d, np.nan])
            else:
                cost = d[0, 0]
                likelihood_sum = d[:, 1].sum()
                map_data.append([seed, cost, likelihood_sum, len(d)])
        cost_data[model] = np.array(map_data)
    return cost_data


def set_parameters(args, plot=False):
    likelihood_data = np.genfromtxt(Path(args.save_dir) / 'likelihood_stepwise.txt', delimiter=',')
    if args.stepwise_eval:
        likelihood_data = likelihood_data[:, 1]
        args.threshold = np.percentile(-np.log(likelihood_data), 82)
        args.prior_num_samples = 1000
        args.prior_nll = np.mean(likelihood_data) * args.prior_num_samples
        if plot:
            plt.hist(likelihood_data, bins=np.arange(0, 0.025, 0.001))
            th = np.exp(-args.threshold)
            plt.axvline(x=th, color='tab:orange')
            plt.title(f'stepwise_likelihood_training {th=:.4f}')
            plt.show()
    else:
        seeds = np.unique(likelihood_data[:, 0])
        avg_likelihood = []
        for s in seeds:
            idx = np.where(likelihood_data[:, 0] == s)[0]
            avg_likelihood.append([s, likelihood_data[:, 1][idx].mean()])
        likelihood_data = np.array(avg_likelihood)
        likelihood_data = -np.log(likelihood_data[:, 1])
        args.threshold = np.percentile(likelihood_data, 90)
        args.prior_num_samples = 10
        args.prior_nll = np.mean(likelihood_data) * args.prior_num_samples
        if plot:
            plt.hist(likelihood_data, bins=np.arange(0, 10, 0.1))
            th = args.threshold
            plt.axvline(x=th, color='tab:orange')
            plt.title(f'mapwise_likelihood_training {th=:.4f}')
            plt.show()


def plot_likelihood(cost_data, model, args):
    seed_data = cost_data[model]
    likelihood = []
    for seed, d in seed_data:
        likelihood.extend([x for x in d[:, 1]])
    likelihood = np.array(likelihood)
    plt.hist(likelihood, bins=np.arange(0, 0.025, 0.001))
    th = np.exp(-args.threshold)
    plt.axvline(x=th, color='tab:orange')
    plt.title(f'{model} ({th=:.4f})')
    plt.show()


if __name__ == "__main__":
    """See lsp.utils.command_line for a full list of args."""
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--stepwise_eval', action='store_true')
    args = parser.parse_args()

    data_files = ['learned_uncorrupted', 'learned_corrupted', 'non_learned_uncorrupted', 'non_learned_corrupted']
    cost_data = read_data(data_files, args, aggregate=True)
    set_parameters(args)
    # plot_likelihood(cost_data, 'learned_corrupted', args)
    args.no_of_maps = len(cost_data['learned_uncorrupted'])

    models = ['learned', 'non_learned', 'ideal_model_selection', 'our_model_selection']
    dist_shift_proportion = np.arange(0, 1, 0.001)
    results = {m: np.zeros_like(dist_shift_proportion) for m in models}
    switch_data = np.zeros((len(dist_shift_proportion), 2))

    for i, corrupt_pose_prob in enumerate(dist_shift_proportion):
        np.random.seed(i)
        args.corrupt_pose_prob = corrupt_pose_prob
        cost_learned = get_cost_learned(cost_data, args)
        results['learned'][i] = cost_learned
        cost_non_learned = get_cost_non_learned(cost_data, args)
        results['non_learned'][i] = cost_non_learned
        cost_ideal_model_selection = get_cost_ideal_model_selection(cost_data, args)
        results['ideal_model_selection'][i] = cost_ideal_model_selection
        cost_our_model_selection = get_cost_our_model_selection(cost_data, args)
        results['our_model_selection'][i] = cost_our_model_selection
        switch_data[i] = [corrupt_pose_prob, args.switch_idx]

    plt.subplot(211)
    for model, results in results.items():
        plt.plot(dist_shift_proportion, results, label=model)

    plt.xlabel('Proportion of distribution shifted maps')
    plt.ylabel('Average cost of navigation')
    plt.legend()

    plt.subplot(212)
    plt.plot(switch_data[:, 0], switch_data[:, 1])
    plt.xlabel('Proportion of distribution shifted maps')
    plt.ylabel('Number of maps until\nswitching to non-learned model')
    plt.show()
