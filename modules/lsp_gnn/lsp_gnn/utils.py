import os
import math
import glob
import sknw
import torch
import random
import itertools
import numpy as np
import scipy.ndimage
import skimage.measure
from PIL import Image
from torch_geometric.data import Data
from skimage.morphology import skeletonize

import lsp
import learning
from gridmap import planning
from lsp.constants import UNOBSERVED_VAL, FREE_VAL, COLLISION_VAL

COMPRESS_LEVEL = 2


def get_n_frontier_points_nearest_to_centroid(subgoal, n):
    """ Get n points from the sorted frontier points' middle indices
    """
    total_points_on_subgoal = len(subgoal.points[0])
    if total_points_on_subgoal <= n:
        return subgoal.points
    mid = total_points_on_subgoal // 2
    return subgoal.points[:, mid - 1:mid + 1]


def compute_skeleton(partial_map, subgoals):
    """Perfom skeletonization on free+unknown space image
    """
    data = {}
    n = 2  # Set using trial and error
    free_unknown_image = \
        lsp.core.mask_grid_with_frontiers(partial_map, subgoals)
    for subgoal in subgoals:
        points_to_be_opened = \
            get_n_frontier_points_nearest_to_centroid(subgoal, n)
        for idx, _ in enumerate(points_to_be_opened[0]):
            x = points_to_be_opened[0][idx]
            y = points_to_be_opened[1][idx]
            free_unknown_image[x][y] = 0
            if partial_map[x][y] == UNOBSERVED_VAL:
                partial_map[x][y] = 0

    free_unknown_image = free_unknown_image != 1
    sk = skeletonize(free_unknown_image)
    data['untrimmed_graph'] = sknw.build_sknw(sk)
    data['untrimmed_edges'] = [value for value in data['untrimmed_graph'].edges()]
    sk[partial_map == UNOBSERVED_VAL] = 0

    data['graph'] = sknw.build_sknw(sk)
    vertex_data = data['graph'].nodes()
    data['vertex_points'] = np.array([vertex_data[i]['o'] for i in vertex_data])
    data['edges'] = [value for value in data['graph'].edges() if value[0] != value[1]]
    return data


def prepare_input_clean_graph(subgoals, vertex_points, edge_data,
                              new_node_dict, has_updated, semantic_grid,
                              wall_class, observation, robot_pose):
    subgoal_nodes = {}
    # This loop checks if a structural node can be replaced as a subgoal
    # node, the subgoal(s) that are not assigned any nodes gets marked
    # to be processed in the next loop
    non_overlapping_subgoals = []
    for subgoal in subgoals:
        index_pos, possible_node = \
            get_subgoal_node(vertex_points, subgoal, subgoal_nodes)
        # find the index_pos of possible_node in vertex_points
        pn = np.array(possible_node)
        sn = subgoal.get_frontier_point()  # this is an np.array
        possible_node = tuple(possible_node)
        subgoal_node = tuple(sn)
        # Check if there is an existing node exactly on that vertex
        if (pn == sn).all():
            if subgoal.props_set and hasattr(subgoal, "nn_input_data"):
                new_node_dict[possible_node] = subgoal.nn_input_data
                has_updated[index_pos] = 0
                subgoal_nodes[possible_node] = subgoal
                continue
            subgoal.nn_input_data = get_input_data(
                semantic_grid, wall_class, subgoal_node,
                observation, robot_pose)
            new_node_dict[possible_node] = subgoal.nn_input_data
            has_updated[index_pos] = 1
            subgoal_nodes[possible_node] = subgoal
            # no need to update edges
        else:
            non_overlapping_subgoals.append(subgoal)

    # This loop creates a node for the marked subgoal(s) from previous
    # loop and connects them to their nearest structural nodes
    for subgoal in non_overlapping_subgoals:
        index_pos, possible_node = \
            get_subgoal_node(vertex_points, subgoal, subgoal_nodes)
        # find the index_pos of possible_node in vertex_points
        pn = np.array(possible_node)
        sn = subgoal.get_frontier_point()  # this is an np.array
        possible_node = tuple(possible_node)
        subgoal_node = tuple(sn)
        subgoal_node_index = len(vertex_points)
        # add the subgoal vertex on the graph
        vertex_points = np.append(vertex_points, [sn], axis=0)
        # add an edge beteen the subgoal vertex and
        # the nearest structural node
        if index_pos != -1:
            # find the adjacent nodes of index_pos'ed vertex
            adjacent_nodes = get_adjacent_vertices(index_pos, edge_data)

            # if index_pos'ed vertex is pendant, choose its parent
            if len(adjacent_nodes) == 1:
                # if the only adjacent node is not a structural node then skip connecting to it
                vp_t = tuple(vertex_points[adjacent_nodes[0]])
                if vp_t not in subgoal_nodes:
                    index_pos = adjacent_nodes[0]
            edge_data.append(tuple([index_pos, subgoal_node_index]))

        if subgoal.props_set and hasattr(subgoal, "nn_input_data"):
            new_node_dict[subgoal_node] = subgoal.nn_input_data
            has_updated.append(0)
            subgoal_nodes[subgoal_node] = subgoal
            continue
        subgoal.nn_input_data = get_input_data(
            semantic_grid, wall_class, subgoal_node,
            observation, robot_pose)
        new_node_dict[subgoal_node] = subgoal.nn_input_data
        has_updated.append(1)
        subgoal_nodes[subgoal_node] = subgoal

    # This loop prunes any structural node that are too close (see default
    # below) to the subgoal nodes
    for subgoal in subgoals:
        # Get the nearby vertices within certain distance (default is <10)
        neighbors = get_neighbors(vertex_points, subgoal, subgoal_nodes)
        c = 0
        # remove the vertices along with the edges if the neighbor is a pendant vertex
        for neighbor in neighbors:
            # Check if this neighbor is a pendent vertex
            edge_to_remove = None
            neighbor = neighbor - c  # the c is here to offset the idx if a vertex is erased
            adjacent_nodes = get_adjacent_vertices(neighbor, edge_data)

            if len(adjacent_nodes) == 1:
                # 1. Find to remove that edge from the list of edges
                if (adjacent_nodes[0], neighbor) in edge_data:
                    edge_to_remove = tuple([adjacent_nodes[0], neighbor])
                elif (neighbor, adjacent_nodes[0]) in edge_data:
                    edge_to_remove = tuple([neighbor, adjacent_nodes[0]])
                # 2. Find to remove that vertex from the list of vertices
                vertex_points_to_remove = neighbor
            if edge_to_remove:
                c += 1
                edge_data.remove(edge_to_remove)
                has_updated.pop(vertex_points_to_remove)
                vertex_points = np.delete(vertex_points, vertex_points_to_remove, axis=0)
                refined_edge_data = []
                for edge_pair in edge_data:
                    t0 , t1 = None, None
                    if edge_pair[0] > vertex_points_to_remove:
                        t0 = edge_pair[0] - 1
                    else:
                        t0 = edge_pair[0]
                    if edge_pair[1] > vertex_points_to_remove:
                        t1 = edge_pair[1] - 1
                    else:
                        t1 = edge_pair[1]
                    refined_edge_data.append(tuple([t0, t1]))
                edge_data = refined_edge_data

    # This loop prunes the structural nodes with 2 degree except, it is the
    # only structural node in the graph
    for subgoal_vertex in subgoal_nodes:
        idx = np.where((vertex_points == subgoal_vertex).all(axis=1))[0][0]
        # find the structural node index that the subgoal node is connected to
        temp_parent = get_adjacent_vertices(idx, edge_data)
        # Handling the case when no parent is found by continuing
        if len(temp_parent) != 1:
            continue
        else:
            parent = temp_parent[0]

        # next find the adjacent nodes to this parent
        adjacent_nodes = get_adjacent_vertices(parent, edge_data)

        # if parent node has degree 2, remove parent and connect the subgoal vertex to
        # the other node connected with parent of the subgoal vertex creating direct link
        while len(adjacent_nodes) == 2:
            new_parent = adjacent_nodes[0] if adjacent_nodes[0] != idx else adjacent_nodes[1]

            # if the new_parent is not a structural node then skip connecting to it
            vp_t = tuple(vertex_points[new_parent])
            if vp_t in subgoal_nodes:
                break

            # 1. Find to remove that edge from the list of edges
            if (new_parent, parent) in edge_data:
                edge_data.remove(tuple([new_parent, parent]))
            elif (parent, new_parent) in edge_data:
                edge_data.remove(tuple([parent, new_parent]))
            if (idx, parent) in edge_data:
                edge_data.remove(tuple([idx, parent]))
            elif (parent, idx) in edge_data:
                edge_data.remove(tuple([parent, idx]))
            edge_data.append(tuple([new_parent, idx]))
            # 2. Find to remove that vertex from the list of vertices
            vertex_points_to_remove = parent
            has_updated.pop(vertex_points_to_remove)
            vertex_points = np.delete(vertex_points, vertex_points_to_remove, axis=0)
            refined_edge_data = []
            for edge_pair in edge_data:
                t0 , t1 = None, None
                if edge_pair[0] > vertex_points_to_remove:
                    t0 = edge_pair[0] - 1
                else:
                    t0 = edge_pair[0]
                if edge_pair[1] > vertex_points_to_remove:
                    t1 = edge_pair[1] - 1
                else:
                    t1 = edge_pair[1]
                refined_edge_data.append(tuple([t0, t1]))
            edge_data = refined_edge_data
            if vertex_points_to_remove < new_parent:
                parent = new_parent - 1
            else:
                parent = new_parent
            if vertex_points_to_remove < idx:
                idx -= 1
            adjacent_nodes = get_adjacent_vertices(parent, edge_data)

    return {
        'subgoal_nodes': subgoal_nodes,
        'vertex_points': vertex_points,
        'edge_data': edge_data,
        'new_node_dict': new_node_dict,
        'has_updated': has_updated
    }


def calculate_euclidian_distance(node, subgoal):
    return ((node[0] - subgoal[0])**2 + (node[1] - subgoal[1])**2)**.5


def get_subgoal_node(vertex_points, subgoal, subgoal_nodes=None):
    possible_node = 0
    index = -1
    distance = 10000
    subgoal_centroid = subgoal.get_frontier_point()
    for idx, node in enumerate(vertex_points):
        m = tuple(node)
        if subgoal_nodes is not None and m in subgoal_nodes:
            continue
        d = calculate_euclidian_distance(node, subgoal_centroid)
        if d < distance:
            distance = d
            possible_node = node
            index = idx
    return index, possible_node


def get_neighbors(vertex_points, subgoal, subgoal_nodes):
    index = []
    distance = 10
    subgoal_centroid = subgoal.get_frontier_point()
    # subgoal_centroid = subgoal.get_centroid()
    for idx, node in enumerate(vertex_points):
        m = tuple(node)
        if m in subgoal_nodes:
            continue
        d = calculate_euclidian_distance(node, subgoal_centroid)
        if d < distance:
            index.append(idx)
    return index


def get_adjacent_vertices(vertex, edge_data):
    """ This method returns the adjacent node(s) from a given node within
    one edge distance depart
    """
    adjacent_nodes_idx = []
    for edge_pair in edge_data:
        if vertex in edge_pair:
            idx = edge_pair[0] if edge_pair[0] != vertex else edge_pair[1]
            adjacent_nodes_idx.append(idx)
    return adjacent_nodes_idx


def preprocess_cnn_data(datum, fn=None, idx=None, args=None):
    data = datum.copy()
    if fn is None:
        if args.input_type == 'image':
            ds_img_list = []
            down_sampled_img = downsample_image(np.transpose(datum['image'][idx], (2, 0, 1)))
            if datum['is_subgoal'][idx] == 1:
                down_sampled_img = down_sampled_img[:, :, 56:72]
                ds_img_list.append(down_sampled_img)
            else:
                _, _, offset = down_sampled_img.shape
                offset /= 8
                offset = int(offset)
                for patch_idx in range(8):
                    t_down_sampled_img = down_sampled_img[:, :, patch_idx * offset:(patch_idx + 1) * offset]
                    ds_img_list.append(t_down_sampled_img)
            data['image'] = torch.tensor(
                (np.array(ds_img_list).astype(np.float32) / 255), dtype=torch.float)
        elif args.input_type == 'seg_image':
            ds_img_list = []
            down_sampled_img = np.transpose(
                datum['image'][idx], (2, 0, 1))[:, ::4, ::4]
            if datum['is_subgoal'][idx] == 1:
                down_sampled_img = down_sampled_img[:, :, 56:72]
                image = convert_seg_image_to_one_hot_coded_vector(down_sampled_img)
                ds_img_list.append(image)
            else:
                _, _, offset = down_sampled_img.shape
                offset /= 8
                offset = int(offset)
                for patch_idx in range(8):
                    t_down_sampled_img = down_sampled_img[:, :, patch_idx * offset:(patch_idx + 1) * offset]
                    image = convert_seg_image_to_one_hot_coded_vector(t_down_sampled_img)
                    ds_img_list.append(image)
            data['image'] = torch.tensor(np.array(
                ds_img_list), dtype=torch.float)
    else:
        data['image'] = torch.cat([
            # Do the preprocessing for running clip encoder
            fn(Image.fromarray(image, mode='RGB')).unsqueeze(0)
            for idx, image in enumerate(data['image'])
        ])
    return data


def preprocess_cnn_eval_data(datum, args=None):
    if args.input_type == 'image':
        down_sampled_img = downsample_image(np.transpose(
            datum['image'], (2, 0, 1)))
        down_sampled_img = down_sampled_img[:, :, 56:72]
        return {
            'image': torch.tensor(np.expand_dims(
                (down_sampled_img.astype(np.float32) / 255),
                axis=0), dtype=torch.float)
        }
    elif args.input_type == 'seg_image':
        down_sampled_img = np.transpose(
            datum['seg_image'], (2, 0, 1))[:, ::4, ::4]
        image = down_sampled_img[:, :, 56:72]
        image = convert_seg_image_to_one_hot_coded_vector(image)
        return {
            'image': torch.tensor(np.expand_dims(
                image, axis=0), dtype=torch.float)
        }


def downsample_image(image, downsample_factor=2):
    return skimage.measure.block_reduce(
        image, (1, 2**downsample_factor, 2**downsample_factor),
        np.average)


def preprocess_ae_img(datum):
    data = {}
    length = len(datum['image'])
    idx = random.randint(0, length - 1)
    image = downsample_image(np.transpose(datum['image'][idx], (2, 0, 1)))
    if datum['is_subgoal'][idx] == 1:
        # take the middle 1/8 of the image:
        image = image[:, :, 56:72]
    else:
        # take any 1/8 of the image
        _, _, offset = image.shape
        offset /= 8
        offset = int(offset)
        patch_idx = random.randint(0, 7)
        image = image[:, :, patch_idx * offset:(patch_idx + 1) * offset]
    data['image'] = torch.tensor((image / 255), dtype=torch.float)
    data['is_feasible'] = torch.tensor(datum['is_feasible'][idx], dtype=torch.float)
    data['delta_success_cost'] = torch.tensor(datum['delta_success_cost'][idx], dtype=torch.float)
    data['exploration_cost'] = torch.tensor(datum['exploration_cost'][idx], dtype=torch.float)
    return data


def preprocess_ae_seg_img(datum):
    data = {}
    length = len(datum['seg_image'])
    idx = random.randint(0, length - 1)
    image = np.transpose(datum['seg_image'][idx], (2, 0, 1))[:, ::4, ::4]
    if datum['is_subgoal'][idx] == 1:
        # take the middle 1/8 of the image:
        image = image[:, :, 56:72]
    else:
        # take any 1/8 of the image
        _, _, offset = image.shape
        offset /= 8
        offset = int(offset)
        patch_idx = random.randint(0, 7)
        image = image[:, :, patch_idx * offset:(patch_idx + 1) * offset]
    image = convert_seg_image_to_one_hot_coded_vector(image)
    data['image'] = torch.tensor(image, dtype=torch.float)
    data['is_feasible'] = torch.tensor(datum['is_feasible'][idx], dtype=torch.float)
    data['delta_success_cost'] = torch.tensor(datum['delta_success_cost'][idx], dtype=torch.float)
    data['exploration_cost'] = torch.tensor(datum['exploration_cost'][idx], dtype=torch.float)
    return data


def convert_seg_image_to_one_hot_coded_vector(image):
    red = np.expand_dims(image[0] == 255, axis=0).astype(float)
    blue = np.expand_dims(image[2] == 255, axis=0).astype(float)
    hallway = np.expand_dims(image[1] == 127, axis=0).astype(float)
    new_image = np.concatenate(
        [red, blue, hallway], axis=0)
    none_of_above = 1 - new_image.max(axis=0)
    none_of_above = np.expand_dims(none_of_above, axis=0).astype(float)
    vectored_image = np.concatenate([new_image, none_of_above], axis=0)
    return vectored_image


def preprocess_gcn_data(datum):
    data = datum.copy()
    temp = [[x[0], x[1]] for x in data['edge_data'] if x[0] != x[1]]
    data['edge_data'] = torch.tensor(list(zip(*temp)), dtype=torch.long)
    data['edge_features'] = torch.tensor(data['edge_features'], dtype=torch.float)
    data['history'] = torch.tensor(data['history'], dtype=torch.long)
    data['goal_distance'] = torch.tensor(data['goal_distance'], dtype=torch.float)
    data['is_subgoal'] = torch.tensor(data['is_subgoal'],
                                      dtype=torch.long)
    return data


def add_super_node(data, input_type):
    ''' This method adds goal as super node and edges to all nodes from it,
    to the graph data. Is used during model training.
    '''
    goal_node_idx = len(data['history'])
    if input_type == 'image':
        data['image'].append(np.zeros((128, 512, 3)))  # Add blank img
    elif input_type == 'seg_image':
        data['seg_image'].append(np.zeros((128, 512, 3)))
    elif input_type == 'wall_class':
        data['wall_class'].append([0, 0, 0])
    # We use the history vector to set only super node as 1 and rest of the graph
    # node as 0
    data['history'] = [0] * len(data['history'])
    data['history'].append(1)

    data['is_subgoal'].append(0)
    data['has_updated'].append(0)
    data['is_feasible'].append(0)
    data['delta_success_cost'].append(0)
    data['exploration_cost'].append(0)
    data['positive_weighting'].append(0)
    data['negative_weighting'].append(0)
    old_edges = [edge_pair for edge_pair in data['edge_data']]
    new_edges = [(idx, goal_node_idx) for idx in range(goal_node_idx)]
    updated_edges = old_edges + new_edges
    # Add feature for each new edges connected to the super node
    for distance in data['goal_distance']:
        feature_vector = []
        feature_vector.append(distance)
        data['edge_features'].append(feature_vector)
    data['edge_data'] = updated_edges
    data['goal_distance'].append(0)
    return data


def get_edge_features(edge_data, vertex_points, node_dict):
    ''' Iterate over the edges to get the edge feature for each edge, right now
    only using squared distance
    '''
    edge_features = []
    for edge in edge_data:
        feature_vector = []
        node1_idx = edge[0]
        node2_idx = edge[1]
        x1 = node_dict[tuple(vertex_points[node1_idx])]['x']
        y1 = node_dict[tuple(vertex_points[node1_idx])]['y']
        x2 = node_dict[tuple(vertex_points[node2_idx])]['x']
        y2 = node_dict[tuple(vertex_points[node2_idx])]['y']
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        feature_vector.append(length)
        edge_features.append(feature_vector)
    return edge_features


def preprocess_gcn_training_data(option=None, fn=None, args=None):
    ''' This method preprocesses the data for GCN training with two options
    marginal history and random history
    '''
    if option is None:
        option = ['marginal', 'random']

    def make_graph(data):
        data = add_super_node(data, args.input_type)
        flag = random.choice(option)

        if args.input_type == 'image':
            data['image'] = args.latent_features_net({
                'image': data['image'],
                'is_subgoal': data['is_subgoal']
            })
        elif args.input_type == 'seg_image':
            data['image'] = args.latent_features_net({
                'image': data['seg_image'],
                'is_subgoal': data['is_subgoal']
            })
        elif args.input_type == 'wall_class':
            data['image'] = torch.tensor(data['wall_class'], dtype=torch.float)
        elif args.use_clip:
            data['image'] = torch.cat([
                # Do the preprocessing for running clip encoder
                fn(Image.fromarray(image, mode='RGB')).unsqueeze(0)
                for idx, image in enumerate(data['image'])
            ])

        # temp_wall_class = [vector[fov] for vector in data['wall_class']]
        # data['wall_class'] = torch.tensor(temp_wall_class, dtype=torch.float)
        temp = [[x[0], x[1]] for x in data['edge_data'] if x[0] != x[1]]
        data['edge_data'] = torch.tensor(list(zip(*temp)), dtype=torch.long)
        data['edge_features'] = torch.tensor(data['edge_features'], dtype=torch.float)

        history = data['history'].copy()
        data['history'] = torch.tensor(data['history'], dtype=torch.long)
        data['is_subgoal'] = torch.tensor(data['is_subgoal'], dtype=torch.long)
        data['has_updated'] = torch.tensor(data['has_updated'], dtype=torch.long)
        data['goal_distance'] = torch.tensor(data['goal_distance'], dtype=torch.float)

        label = data['is_feasible'].copy()
        data['is_feasible'] = torch.tensor(data['is_feasible'], dtype=torch.float)
        data['delta_success_cost'] = torch.tensor(
            data['delta_success_cost'], dtype=torch.float)
        data['exploration_cost'] = torch.tensor(
            data['exploration_cost'], dtype=torch.float)
        data['positive_weighting'] = torch.tensor(
            data['positive_weighting'], dtype=torch.float)
        data['negative_weighting'] = torch.tensor(
            data['negative_weighting'], dtype=torch.float)
        tg_GCN_format = Data(x=data['image'],
                             edge_index=data['edge_data'],
                             edge_features=data['edge_features'],
                             is_subgoal=data['is_subgoal'],
                             has_updated=data['has_updated'],
                             goal_distance=data['goal_distance'],
                             y=data['is_feasible'],
                             dsc=data['delta_success_cost'],
                             ec=data['exploration_cost'],
                             pweight=data['positive_weighting'],
                             nweight=data['negative_weighting'])

        if flag == 'marginal':
            # Formating for training only with marginal history vector
            tg_GCN_format.__setitem__('history', data['history'])
        elif flag == 'random':
            # Formating for training with randomly chosen
            # history vector
            history_vector = generate_random_history_combination(
                history, label)
            data['history'] = torch.tensor(history_vector,
                                           dtype=torch.long)
            tg_GCN_format.__setitem__('history', data['history'])
        result = tg_GCN_format
        return result
    return make_graph


def generate_random_history_combination(history, node_labels):
    pool = [i for i, val in enumerate(history)
            if val == 1 and node_labels[i] == 0]
    random_history = node_labels.copy()
    n = history.count(1)
    c = node_labels.count(1)
    x = random.randint(0, n - c)
    for _ in range(x):
        sub = random.choice(pool)
        random_history[sub] = 1
        pool.remove(sub)
    return random_history


def generate_all_rollout_history(history):
    pool = [i for i, val in enumerate(history)
            if val == 1]
    n = history.count(1)
    history_vectors = []
    for idx in range(n):
        combis = itertools.combinations(pool, idx)
        for a_tuple in combis:
            temp = history.copy()
            for val in a_tuple:
                temp[val] = 0
            history_vectors.append(temp)
    return history_vectors


def write_datum_to_file(args, datum, counter):
    """Write a single datum to file and append name to csv record."""
    # Get the data file name
    data_filename = os.path.join('pickles', f'dat_{args.current_seed}_{counter}.pgz')
    learning.data.write_compressed_pickle(
        os.path.join(args.save_dir, data_filename), datum)
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    with open(os.path.join(args.save_dir, csv_filename), 'a') as f:
        f.write(f'{data_filename}\n')


def get_data_path_names(args):
    training_data_files = glob.glob(os.path.join(args.data_csv_dir, "*train*.csv"))
    testing_data_files = glob.glob(os.path.join(args.data_csv_dir, "*test*.csv"))
    return training_data_files, testing_data_files


def get_input_data(semantic_grid, wall_class, vertex_point,
                   observation=None, robot_pose=None):
    x, y = vertex_point
    x, y = int(x), int(y)
    if semantic_grid[x][y] == wall_class['red']:
        one_hot_coded_vector = [1, 0, 0]
    elif semantic_grid[x][y] == wall_class['blue']:
        one_hot_coded_vector = [0, 1, 0]
    else:
        one_hot_coded_vector = [0, 0, 1]

    return {'image': None,
            'seg_image': None,
            'input': one_hot_coded_vector,
            'x': x,
            'y': y,
            }


def check_if_same(new_data, old_data):
    '''
    This method checks if the last saved data and the current data are the same
    or not.
    '''
    if new_data['vertex_points'].shape[0] == old_data['vertex_points'].shape[0]:
        is_same_vertices = (new_data['vertex_points'] == old_data['vertex_points']).all()
    else:
        is_same_vertices = False
    is_same_edges = set(new_data['edge_data']) == set(old_data['edge_data'])
    return is_same_vertices and is_same_edges


def check_if_reachable(
        known_grid, grid, goal_pose, start_pose,
        frontier, downsample_factor=1):
    '''
    Checks if an alternate subgoal path is still reachable. This changes the
    label for the subgoal whose alternate path has already explored the merging
    point through this subgoal
    '''
    inflated_mixed_grid = np.ones_like(known_grid)
    inflated_mixed_grid[np.logical_and(
        known_grid == FREE_VAL,
        grid == UNOBSERVED_VAL)] = UNOBSERVED_VAL
    occupancy_grid = np.copy(inflated_mixed_grid)
    occupancy_grid[occupancy_grid == FREE_VAL] = COLLISION_VAL
    occupancy_grid[occupancy_grid == UNOBSERVED_VAL] = FREE_VAL
    if downsample_factor > 1:
        occupancy_grid = skimage.measure.block_reduce(
            occupancy_grid, (downsample_factor, downsample_factor), np.min)

    # Compute the cost grid
    cost_grid = planning.compute_cost_grid_from_position(
        occupancy_grid,
        start=[
            goal_pose.x // downsample_factor, goal_pose.y // downsample_factor
        ],
        use_soft_cost=False,
        only_return_cost_grid=True)

    # Compute the cost for each frontier
    fpts = frontier.points // downsample_factor
    cost = downsample_factor * (cost_grid[fpts[0, :], fpts[1, :]].min())

    if math.isinf(cost):
        cost = 100000000000
        known_cost_grid = planning.compute_cost_grid_from_position(
            known_grid,
            start=[
                start_pose.x // downsample_factor,
                start_pose.y // downsample_factor
            ],
            only_return_cost_grid=True)

        unk_regions = (inflated_mixed_grid == UNOBSERVED_VAL)
        labels, nb = scipy.ndimage.label(unk_regions)
        fp = frontier.points // downsample_factor
        flabel = labels[fp[0, :], fp[1, :]].max()
        cost_region = known_cost_grid[labels == flabel]
        min_cost = cost_region.min()
        max_cost = cost_region.max()
        if min_cost > 1e8:
            frontier.set_props(prob_feasible=0.0)
            frontier.is_obstructed = True
        else:
            if max_cost > 1e8:
                cost_region[cost_region > 1e8] = 0
                max_cost = cost_region.max()

            exploration_cost = 2 * downsample_factor * (max_cost -
                                                        min_cost)
            frontier.set_props(prob_feasible=0.0,
                               exploration_cost=exploration_cost)
        # frontier.set_props(prob_feasible=0.0, is_obstructed=True)
        frontier.just_set = False

    if frontier.prob_feasible == 1.0:
        return True
    return False


def parse_args():
    parser = lsp.utils.command_line.get_parser()
    parser.add_argument('--relative_positive_weight',
                        default=1.0,
                        help='Initial learning rate',
                        type=float)
    parser.add_argument('--train_cnn_lsp', action='store_true')
    parser.add_argument('--train_marginal_lsp', action='store_true')
    parser.add_argument('--fov', type=int, default=None,)
    parser.add_argument(
        '--input_type',
        type=str,
        required=False)
    parser.add_argument(
        '--epoch_size',
        type=int,
        required=False,
        default=10000,
        help='The number of steps in epoch. total_input_count / batch_size')
    parser.add_argument(
        '--current_seed',
        type=int)
    parser.add_argument(
        '--data_file_base_name',
        type=str,
        required=False)
    parser.add_argument('--logfile_name', type=str, default='logfile.txt')
    parser.add_argument(
        '--learning_rate_decay_factor',
        default=0.5,
        help='How much learning rate decreases between epochs.',
        type=float)
    group = parser.add_argument_group('Make Training Data Arguments')
    group.add_argument(
        '--lsp_weight',
        type=float,
        required=False,
        help='Set the weight of LSP loss contribution during AE training')
    group.add_argument(
        '--loc_weight',
        type=float,
        required=False,
        help='Set the weight of location loss contribution during AE training')
    group.add_argument(
        '--loss',
        type=str,
        required=False,
        help='Set the l1 or l2 loss for image loss calucation')
    group.add_argument(
        '--network_file',
        type=str,
        required=False,
        help='Directory with the name of the conditional gcn model')
    group.add_argument(
        '--autoencoder_network_file',
        type=str,
        required=False,
        help='Directory with the name of the autoencoder model')
    group.add_argument(
        '--clip_network_file',
        type=str,
        required=False,
        help='Directory with the name of the clip(by open.ai) encoder model')
    group.add_argument(
        '--cnn_network_file',
        type=str,
        required=False,
        help='Directory with the name of the base/cnn lsp model')
    group.add_argument(
        '--gcn_network_file',
        type=str,
        required=False,
        help='Directory with the name of the marginal gcn model')
    group.add_argument(
        '--image_filename',
        type=str,
        required=False,
        help='File name for the completed evaluations')
    group.add_argument(
        '--data_csv_dir',
        type=str,
        required=False,
        help='Directory in which to save the data csv')
    group.add_argument(
        '--pickle_directory',
        type=str,
        required=False,
        help='Directory in which to save the pickle dataums')
    group.add_argument(
        '--csv_basename',
        type=str,
        required=False,
        help='Directory in which to save the CSV base file')
    group = parser.add_argument_group('Neural Network Training Testing \
        Arguments')
    group.add_argument(
        '--core_directory',
        type=str,
        required=False,
        help='Directory in which to look for data')
    group.add_argument(
        '--num_training_elements',
        type=int,
        required=False,
        default=5000,
        help='Number of training samples')
    group.add_argument(
        '--num_testing_elements',
        type=int,
        required=False,
        default=1000,
        help='Number of testing samples')
    group.add_argument(
        '--num_steps',
        type=int,
        required=False,
        default=10000,
        help='Number of steps while iterating')
    group.add_argument(
        '--test_log_frequency',
        type=int,
        required=False,
        default=10,
        help='Frequecy of testing log to be generated')
    group.add_argument(
        '--learning_rate',
        type=float,
        required=False,
        default=.001,
        help='Learning rate of the model')
    group.add_argument('--experiment_name', type=str, default='raihan_v1')

    return parser.parse_args()
