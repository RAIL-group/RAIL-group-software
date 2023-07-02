import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import re
import scipy.stats


def process_results_data(args):
    # Load data and loop through rows
    datasets = []
    if args.do_intervene:
        for data_file in args.data_file:
            datasets.append(_process_data_file_intervene(data_file))
    else:
        for data_file in args.data_file:
            datasets.append(_process_data_file_base(data_file))

    return datasets


def _process_data_file_base(data_file):
    data = []
    for line in open(data_file).readlines():
        d = re.match(
            r'\[Learn\]  s: (.*?) . learned: (.*?) . baseline: (.*?)\n', line)
        if d is not None:
            d = d.groups()
            data.append([int(d[0]), float(d[1]), float(d[2])])
        else:
            d = re.match(
                r'\[Learn\] \[ERR\] s: (.*?) . learned: (.*?) . baseline: (.*?)\n',
                line)
            if d is None:
                continue
            d = d.groups()
            dm = min(float(d[1]), float(d[2]))
            data.append([int(d[0]), float(dm), float(dm)])

    return pd.DataFrame(data,
                        columns=['seed', 'learned_cost', 'baseline_cost'])


def _process_data_file_intervene(data_file):
    data = []
    for line in open(data_file).readlines():
        d = re.match(
            r'\[Learn\]  s: (.*?) . learned: (.*?) . intervene: (.*?) . baseline: (.*?)\n',
            line)
        if d is not None:
            d = d.groups()
            data.append(
                [int(d[0]), True,
                 float(d[1]),
                 float(d[2]),
                 float(d[3])])
        else:
            d = re.match(
                r'\[Learn\] \[ERR\] s: (.*?) . learned: (.*?) . intervene: (.*?) . baseline: (.*?)\n',
                line)
            if d is None:
                continue
            d = d.groups()
            data.append(
                [int(d[0]), False,
                 float(d[1]),
                 float(d[1]),
                 float(d[3])])

    return pd.DataFrame(data,
                        columns=[
                            'seed', 'did_succeed', 'learned_cost',
                            'intervene_cost', 'baseline_cost'
                        ])


def build_plot(fig, data, args):
    ax = plt.subplot(221)
    ax.scatter(data['seed'],
               data['learned_cost'] - data['baseline_cost'],
               c=data['seed'])
    ax.plot(
        data['seed'],
        pd.DataFrame(data['learned_cost'] -
                     data['baseline_cost']).rolling(25).mean().to_numpy(), 'y')
    ax.plot(
        data['seed'],
        pd.DataFrame(data['learned_cost'] -
                     data['baseline_cost']).rolling(50).mean().to_numpy(), 'c')
    ax.plot(
        data['seed'],
        pd.DataFrame(data['learned_cost'] - data['baseline_cost']).rolling(
            100, min_periods=1).mean().to_numpy(), 'b')
    ax.plot(data['seed'], 0 * data['seed'], 'k')
    ax = plt.subplot(223)
    ax.scatter(data['seed'],
               np.log(data['learned_cost']) - np.log(data['baseline_cost']),
               c=data['seed'])
    ax.plot(data['seed'], 0 * data['seed'], 'k')
    ax = plt.subplot(122)
    ax.scatter(data['baseline_cost'], data['learned_cost'], c=data['seed'])
    ax.set_aspect('equal', adjustable='box')
    cb = min(max(data['baseline_cost']), max(data['learned_cost']))
    ax.plot([0, cb], [0, cb], 'k')

    # Print some additional statistics
    data['cost_diff'] = data['learned_cost'] - data['baseline_cost']
    data = data.sort_values(by='cost_diff', ascending=False)
    print(data['seed'][:50].to_numpy())


def build_density_scatterplot(ax,
                              data,
                              bounds=None,
                              cmap='viridis',
                              title=None):
    xy = np.vstack([data['baseline_cost'], data['learned_cost']])
    z = scipy.stats.gaussian_kde(xy)(xy)

    data['zs'] = z
    data = data.sort_values(by=['zs'])
    z = data['zs']
    colors = cm.get_cmap(cmap)((z - z.min()) / (z.max() - z.min()) * 0.75 +
                               0.25)
    axins = ax.inset_axes([-0.02, 0.64, 0.38, 0.38])

    def add_plot(ax, bounds, is_inset=False, sm_bounds=None):
        if is_inset:
            size = 4
        else:
            size = 8

        ax.scatter(data['baseline_cost'],
                   data['learned_cost'],
                   c=colors,
                   s=size)
        ax.set_aspect('equal', adjustable='box')

        cb = max(bounds) * 1.1
        ax.plot([0, cb], [0, cb], 'k:')
        ax.set_xlim([0, cb])
        ax.set_ylim([0, cb])
        if is_inset:
            ax.set_xticks([max(bounds)])
            ax.set_yticks([max(bounds)])
            import matplotlib.patches
            rect = matplotlib.patches.Rectangle((0, 0),
                                                max(sm_bounds),
                                                max(sm_bounds),
                                                edgecolor='gray',
                                                fill=False)
            ax.add_patch(rect)

        else:
            ax.set_yticks(range(0, 81, 20))
            ax.set_title(title)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.0)
                ax.spines[axis].set_color('gray')

    add_plot(ax, bounds, is_inset=False)
    add_plot(axins, [400], is_inset=True, sm_bounds=np.array(bounds) * 1.1)


def build_plot_comparison(fig, datasets, args):
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    import itertools
    marker = itertools.cycle(('o', '*', '.', 'x'))

    for data, name in zip(datasets, args.data_file):
        d_cost = data['learned_cost'] - data['baseline_cost']

        line = ax2.plot(
            data['seed'],
            pd.DataFrame(d_cost).rolling(100, min_periods=1).mean().to_numpy())
        ax2.plot(data['seed'], 0 * data['seed'], 'k')

        # print(line[-1].get_color())
        print(f"\nDataset: {name}")
        print(data.describe())
        ax1.scatter(data['seed'],
                    d_cost,
                    color=line[-1].get_color(),
                    s=4,
                    marker=next(marker))
        print(
            f"  Ratio of Means: {data['learned_cost'].mean()/data['baseline_cost'].mean()}"
        )
        print(
            f"  Mean of Ratios: {(data['learned_cost']/data['baseline_cost']).mean()}"
        )

    ax1.legend(args.data_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file',
                        type=str,
                        nargs='+',
                        required=False,
                        default=None)
    parser.add_argument('--output_image_file',
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument('--xpassthrough',
                        type=str,
                        required=False,
                        default='false')
    parser.add_argument('--do_intervene', action='store_true')
    args = parser.parse_args()
    data_all = process_results_data(args)
    for d in data_all:
        # Resolution is 0.1, so divide all costs by 10
        d['learned_cost'] /= 10
        d['baseline_cost'] /= 10
        try:
            d['intervene_cost'] /= 10
        except KeyError:
            pass

    if not args.do_intervene:
        print("\nResults Overview:")
        for d, name in zip(data_all, args.data_file):
            print(name)
            print(f"  Mean Learned Cost: {d.learned_cost.mean()}")
            print(f"  Total Learned Cost: {d.learned_cost.sum()}")
        print("Non-Learned")
        print(f"  Mean Non-Learned Cost: {d.baseline_cost.mean()}")
        print(f"  Total Non-Learned Cost: {d.baseline_cost.sum()}")
    else:
        for d, name in zip(data_all, args.data_file):
            print("")
            print(f"Dataset: {name}")
            print(f"  Num Successful: {sum(d.did_succeed.astype(int))}")
            print(f"  Learned Cost Total: {sum(d.learned_cost)}")
            print(f"  Intervene Cost Total: {sum(d.intervene_cost)}")
            print(f"  Baseline Cost Total: {sum(d.baseline_cost)}")

    if len(data_all) == 3:
        num_files = 3
        fig = plt.figure(figsize=(3 * num_files, 3), dpi=300)
        bounds = [
            max([d['baseline_cost'].max() for d in data_all]),
            max([d['learned_cost'].max() for d in data_all])
        ]
        bounds = [125]

        cmaps = ['Reds', 'Oranges', 'Blues']
        titles = ['All Subgoal Props', '4 Subgoal Props', 'No Subgoal Props']

        for ii, (data, cmap, title) in enumerate(zip(data_all, cmaps, titles)):
            ax = plt.subplot(1, num_files, ii + 1)
            build_density_scatterplot(ax,
                                      data,
                                      bounds=bounds,
                                      cmap=cmap,
                                      title=title)

        plt.savefig(args.output_image_file)
