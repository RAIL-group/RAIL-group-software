import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import exists

import lsp_cond
import learning

viridis = cm.get_cmap('viridis')


def test_lsp_cond_inspect_graph_datum(do_debug_plot, unity_path):
    ''' Visualize as a graph
    '''
    pickle_path = '/data/lsp_cond/pickles/dat_1_17.pgz'
    if not exists(pickle_path):
        print("Pickle file path does not exist")
        return True
    datum = learning.data.load_compressed_pickle(pickle_path)
    if do_debug_plot:
        plt.ion()
        plt.figure(figsize=(12, 4))
        plt.clf()
        ax = plt.subplot(131)
        lsp_cond.plotting.plot_semantic_grid_with_frontiers(
            ax, datum['observed_map'], datum['known_map'], datum['subgoals'],
            datum['semantic_grid'], datum['semantic_labels']
        )
        lsp_cond.plotting.plot_pose(ax, datum['goal'], color='green', filled=False)

        ax = plt.subplot(132)
        lsp_cond.plotting.plot_grid(
            ax, datum['observed_map'], datum['known_map'], None
        )
        is_subgoal = datum['is_subgoal']
        prob_feasible = datum['is_feasible']
        vertex_points = datum['vertex_points']
        for vp_idx, ps in enumerate(vertex_points):
            if not is_subgoal[vp_idx]:
                color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                plt.plot(ps[0], ps[1], '.', color=color, markersize=3, markeredgecolor='r')
        for vp_idx, ps in enumerate(vertex_points):
            if is_subgoal[vp_idx]:
                color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
        for (start, end) in datum['edge_data']:
            p1 = vertex_points[start]
            p2 = vertex_points[end]
            x_values = [p1[0], p2[0]]
            y_values = [p1[1], p2[1]]
            plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

        ax = plt.subplot(133)
        lsp_cond.plotting.plot_grid(
            ax, datum['observed_map'], datum['known_map'], None)
        for vp_idx, ps in enumerate(vertex_points):
            if is_subgoal[vp_idx]:
                marker = '.'
            else:
                marker = '+'
            x, y = ps
            x, y = int(x), int(y)
            if datum['semantic_grid'][x][y] == datum['semantic_labels']['red']:
                color = 'red'
            elif datum['semantic_grid'][x][y] == datum['semantic_labels']['blue']:
                color = 'blue'
            else:
                color = 'gray'
            plt.plot(ps[0], ps[1], marker, color=color, markersize=3)
        plt.show()
        # image_file = '/data/lsp_conditional/inspected_datum.png'
        # plt.savefig(image_file, dpi=250)


def get_args():
    args = lambda: None  # noqa
    args.base_resolution = .4
    args.inflation_radius_m = .75
    return args


def test_lsp_cond_subgoal_reveal(do_debug_plot, unity_path):
    ''' Visualize as a graph
    '''
    pickle_path = '/data/lsp_cond/pickles/dat_1_17.pgz'
    if not exists(pickle_path):
        print("Pickle file path does not exist")
        return True
    datum = learning.data.load_compressed_pickle(pickle_path)
    args = get_args()
    if do_debug_plot:
        plt.ion()
        plt.figure(figsize=(12, 4))
        plt.clf()
        ax = plt.subplot(131)
        ax.set_title('Underlying symantic grid')
        lsp_cond.plotting.plot_semantic_grid_with_frontiers(
            ax, datum['observed_map'], datum['known_map'], datum['subgoals'],
            datum['semantic_grid'], datum['semantic_labels']
        )
        lsp_cond.plotting.plot_pose(ax, datum['goal'], color='green', filled=False)

        ax = plt.subplot(132)
        ax.set_title('Subgoal masked view')
        lsp_cond.plotting.plot_grid(
            ax, datum['observed_map'], datum['known_map'], None
        )
        is_subgoal = datum['is_subgoal']
        prob_feasible = datum['is_feasible']
        vertex_points = datum['vertex_points']
        for vp_idx, ps in enumerate(vertex_points):
            if not is_subgoal[vp_idx]:
                color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                plt.plot(ps[0], ps[1], '+', color=color, markersize=3, markeredgecolor='r')
        for vp_idx, ps in enumerate(vertex_points):
            if is_subgoal[vp_idx]:
                color = viridis(is_subgoal[vp_idx] * prob_feasible[vp_idx])
                plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
        for (start, end) in datum['edge_data']:
            p1 = vertex_points[start]
            p2 = vertex_points[end]
            x_values = [p1[0], p2[0]]
            y_values = [p1[1], p2[1]]
            plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)

        ax = plt.subplot(133)
        ax.set_title('Subgoal revealed view')

        from common import Pose
        robot_is_here = Pose(x=35, y=65)

        frontiers = [f for f in datum['subgoals']]
        forntier_idx = 0
        frontier_to_reveal = frontiers.pop(forntier_idx)
        
        subgoal_revealed_grid, f_remove_idx = lsp_cond.utils.get_frontier_revealing_grid(
            datum['observed_map'], datum['known_map'], frontier_to_reveal, frontiers
        )

        inflated_grid = lsp_cond.utils.get_inflated_occupancy_grid(
            subgoal_revealed_grid, args.inflation_radius_m / args.base_resolution, robot_is_here
        )

        # Remove the frontiers that get revealed along with the revealed region
        for idx in f_remove_idx[::-1]:
            frontiers.pop(idx)

        uncleaned_graph = lsp_cond.utils.compute_skeleton(inflated_grid, frontiers)
        vertex_points = uncleaned_graph['vertex_points']
        edge_data = uncleaned_graph['edges']
        new_node_dict = {}

        clean_data = lsp_cond.utils.prepare_input_clean_graph(
            frontiers, vertex_points, edge_data,
            new_node_dict, [0] * len(vertex_points), datum['semantic_grid'],
            datum['wall_class'], None, robot_is_here
        )

        lsp_cond.plotting.plot_grid(
            ax, subgoal_revealed_grid, datum['known_map'], None
        )
        vertex_points = clean_data['vertex_points']
        edge_data = clean_data['edge_data']
        for vp_idx, ps in enumerate(vertex_points):
            ps = tuple(ps)
            if ps not in clean_data['subgoal_nodes']:
                color = viridis(0)
                plt.plot(ps[0], ps[1], '+', color=color, markersize=3, markeredgecolor='r')
        for vp_idx, ps in enumerate(vertex_points):
            ps = tuple(ps)
            if ps in clean_data['subgoal_nodes']:
                color = viridis(clean_data['subgoal_nodes'][ps].prob_feasible)
                plt.plot(ps[0], ps[1], '.', color=color, markersize=4)
        for (start, end) in edge_data:
            p1 = vertex_points[start]
            p2 = vertex_points[end]
            x_values = [p1[0], p2[0]]
            y_values = [p1[1], p2[1]]
            plt.plot(x_values, y_values, 'c', linestyle="--", linewidth=0.3)
        image_file = '/data/lsp_cond/revealed_subgoal.png'
        plt.savefig(image_file, dpi=250)
