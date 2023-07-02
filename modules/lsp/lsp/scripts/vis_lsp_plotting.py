import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import re
import scipy.stats
import os


def process_results_data(args):
    # Load data and loop through rows
    data = []
    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . learned: (.*?) . baseline: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1]), float(d[2])])

    return pd.DataFrame(data, columns=['seed', 'learned_cost', 'baseline_cost'])


def build_plot(ax, data, args, cmap='Blues'):
    xy = np.vstack([data['baseline_cost'], data['learned_cost']])
    z = scipy.stats.gaussian_kde(xy)(xy)

    data['zs'] = z
    data = data.sort_values(by=['zs'])
    z = data['zs']
    colors = matplotlib.colormaps[cmap]((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.25)

    ax.scatter(data['baseline_cost'], data['learned_cost'], c=colors)
    ax.set_aspect('equal', adjustable='box')
    cb = 1.05 * max(max(data['baseline_cost']), max(data['learned_cost']))
    ax.plot([0, cb], [0, cb], 'k', alpha=0.3)
    ax.set_xlim([0, cb])
    ax.set_ylim([0, cb])
    ax.set_xlabel('Optimistic Baseline Cost')
    ax.set_ylabel('Learned Planner Cost')


def add_tensorboard_data(ax, folder_path):
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    def get_smoothed_event_data(folder_path, smooth_factor=500, scalar_name='Loss/total_loss'):
        # Get the event file
        fname = sorted(os.listdir(folder_path))[-1]
        epath = os.path.join(folder_path, fname)

        # Load the event file data
        event_acc = EventAccumulator(epath)
        event_acc.Reload()

        # Smooth the data
        data = pd.DataFrame([(e.step, e.value) for e in event_acc.Scalars('Loss/total_loss')])
        data = data.fillna(0)
        smoothed_data = data[1].ewm(halflife=str(smooth_factor),
                                    times=pd.DatetimeIndex(data[0])).mean()
        smoothed_data.fillna(0)

        return {'steps': data[0],
                'unsmoothed': data[1],
                'smoothed': smoothed_data}

    data_train = get_smoothed_event_data(os.path.join(folder_path, 'train'))
    data_test = get_smoothed_event_data(os.path.join(folder_path, 'test'))
    plt.plot(data_train['steps'], data_train['unsmoothed'], color='magenta', alpha=0.1)
    plt.plot(data_test['steps'], data_test['unsmoothed'], 'b', alpha=0.1)
    plt.plot(data_train['steps'], data_train['smoothed'], color='magenta', alpha=0.8)
    plt.plot(data_test['steps'], data_test['smoothed'], 'b', alpha=0.8)
    plt.xlim([data_test['steps'].min(), data_test['steps'].max()])
    plt.ylim(0, data_test['smoothed'][10:].max())

    plt.legend(['Training (unsmoothed)',
                'Testing (unsmoothed)',
                'Training (smoothed)',
                'Testing (smoothed)'])
    plt.title('Neural Net Training Loss')
    plt.xlabel('Steps')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate a figure (and write to file) for results from the interpretability project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file',
                        type=str,
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
    args = parser.parse_args()

    data = process_results_data(args)

    print(data.describe())
    fig = plt.figure(dpi=200, figsize=(9, 6))

    from matplotlib import gridspec
    spec = gridspec.GridSpec(nrows=1, ncols=2,
                             width_ratios=[3, 1.8])

    ax = fig.add_subplot(spec[0])
    build_plot(ax, data, args)
    cost_mean_base = data['baseline_cost'].mean()
    cost_mean_learn = data['learned_cost'].mean()
    improvement = (cost_mean_base - cost_mean_learn) / (cost_mean_base)
    title_string = (f"Results: {args.output_image_file}\n"
                    f"Optimistic Baseline Cost: {cost_mean_base:.2f}\n"
                    f"Learned Cost: {cost_mean_learn:.2f}\n"
                    f"Improvement %: {100*improvement:0.2f} | {data['baseline_cost'].size} seeds")
    ax.set_title(title_string)
    print(title_string)

    try:
        ax = fig.add_subplot(spec[1])
        bpath = args.data_file.replace('/results/', '/training_logs/')
        bpath = os.path.dirname(bpath)
        add_tensorboard_data(ax, bpath)
    except:  # noqa:E722
        print("TensorBoard data could not be loaded.")
        print(args)

    plt.savefig(args.output_image_file, dpi=300)
