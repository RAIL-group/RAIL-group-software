import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy.stats


def process_results_data(args):
    """This function prepares the result for all four planning approach from the logfile"""
    data = []
    for line in open(args.data_file).readlines():
        d = re.match(
            r'.*?s: (.*?) . baseline: (.*?) . naive: (.*?) . lsp: (.*?). gcn_lsp: (.*?)\n',
            line)
        if d is None:
            continue
        d = d.groups()
        data.append(
            [int(d[0]), float(d[1]) * args.base_resolution,
             float(d[2]) * args.base_resolution, float(d[3]) * args.base_resolution,
             float(d[4]) * args.base_resolution])

    return pd.DataFrame(
        data,
        columns=['seed', 'Known', 'Naive', 'CNN_LSP', 'GCN_LSP']
    )


def process_gnn_results_data(args):
    """This function prepares the result for gnn planning approach from the mlsp_logfile"""
    data = []
    for line in open(args.data_file).readlines():
        d = re.match(r'.*?s: (.*?) . gcn_lsp: (.*?)\n', line)
        if d is None:
            continue
        d = d.groups()
        data.append([int(d[0]), float(d[1])])

    return pd.DataFrame(
        data,
        columns=['seed', 'GCN_LSP']
    )


def build_plot(fig, data, args, cmap='Blues'):
    """Function for scatter plot performance with gnn against others"""
    xy = np.vstack([data['Naive'], data['GCN_LSP']])
    z = scipy.stats.gaussian_kde(xy)(xy)

    data['zs'] = z
    data = data.sort_values(by=['zs'])
    z = data['zs']
    colors = matplotlib.colormaps.get_cmap(cmap)((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.25)

    fig.gca()
    ax = plt.subplot(121)
    ax.scatter(data['Naive'], data['GCN_LSP'], c=colors)
    cb = min(max(data['Naive']), max(data['GCN_LSP']))
    ax.axis(xmin=0, ymin=0)
    ax.plot([0, cb], [0, cb], linestyle=':', color='gray')
    ax.set_title("LSP-GNN vs Non-learned baseline")
    ax.set_xlabel("Non-learned baseline expected cost")
    ax.set_ylabel("LSP-GNN expected cost")
    xy = np.vstack([data['CNN_LSP'], data['GCN_LSP']])
    z = scipy.stats.gaussian_kde(xy)(xy)

    data['zs'] = z
    data = data.sort_values(by=['zs'])
    z = data['zs']
    colors = matplotlib.colormaps.get_cmap(cmap)((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.25)
    ax = plt.subplot(122)
    ax.scatter(data['CNN_LSP'], data['GCN_LSP'], c=colors)
    cb = min(max(data['CNN_LSP']), max(data['GCN_LSP']))
    ax.axis(xmin=0, ymin=0)
    # ax.xlim([0, cb])
    ax.plot([0, cb], [0, cb], linestyle=':', color='gray')
    ax.set_title("LSP-GNN vs LSP-Local")
    ax.set_xlabel("LSP-Local expected cost")


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
    parser.add_argument('--gnn', action='store_true')
    parser.add_argument('--base_resolution',
                        type=float,
                        required=True,
                        default='false')
    args = parser.parse_args()

    if args.gnn:
        data = process_gnn_results_data(args)
        print(data.describe())
    else:
        data = process_results_data(args)
        print(data.describe())
        fig = plt.figure(dpi=300, figsize=(10, 5))
        build_plot(fig, data, args)
        plt.savefig(args.output_image_file, dpi=300)
