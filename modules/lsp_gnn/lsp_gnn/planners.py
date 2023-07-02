import copy
import torch
import numpy as np

import lsp
import gridmap
import lsp_gnn
from lsp.planners.planner import Planner
# from lsp_gnn.learning.models.auto_encoder import AutoEncoder
from lsp_gnn.learning.models.gcn import WallClassGNN
from lsp_gnn.learning.models.cnn_lsp import WallClassLSP


NUM_MAX_FRONTIERS = 8


class ConditionalSubgoalPlanner(Planner):
    ''' The parent planner class that takes care of the updated graph representation of
    the environment. '''
    def __init__(self, goal, args, device=None,
                 semantic_grid=None, wall_class=None):
        super(ConditionalSubgoalPlanner, self).__init__(goal)

        self.subgoals = set()
        self.selected_subgoal = None
        self.semantic_grid = semantic_grid
        self.wall_class = wall_class
        self.args = args

        self.vertex_points = None
        self.edge_data = None

        self.inflation_radius = args.inflation_radius_m / args.base_resolution
        if self.inflation_radius >= np.sqrt(5):
            self.downsample_factor = 2
        else:
            self.downsample_factor = 1
        self.verbose = False
        self.old_node_dict = {}

    def update(self, observation, observed_map,
               subgoals, robot_pose, *args, **kwargs):
        """Updates the internal state with the new grid/pose/laser scan.
        This function also computes a few necessary items, like which
        frontiers have recently been updated and computes their properties
        from the known grid.
        """
        self.observation = observation
        self.observed_map = observed_map
        self.robot_pose = robot_pose
        # Store the inflated grid after ensuring that the unreachable 'free
        # space' is set to 'unobserved'. This avoids trying to plan to
        # unreachable space and avoids having to check for this everywhere.
        inflated_grid = self._get_inflated_occupancy_grid()
        self.inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
            inflated_grid, robot_pose)

        # Compute the new frontiers and update stored frontiers
        new_subgoals = set([copy.copy(s) for s in subgoals])
        self.subgoals = lsp.core.update_frontier_set(
            self.subgoals,
            new_subgoals,
            max_dist=2.0 / self.args.base_resolution,  # Was set to 20.0
            chosen_frontier=self.selected_subgoal)
        self.original_subgoal = self.subgoals.copy()
        # Also check that the goal is not inside the frontier
        lsp.core.update_frontiers_goal_in_frontier(self.subgoals,
                                                   self.goal)

        skeletonized_graph_data = lsp_gnn.utils.compute_skeleton(
            inflated_grid.copy(), self.subgoals)
        self.vertex_points = skeletonized_graph_data['vertex_points']
        self.edge_data = skeletonized_graph_data['edges']

        # Update the subgoal inputs & get representative nodes for the subgoals
        self.subgoal_nodes = self._identify_subgoal_nodes()

        # Once the subgoal inputs are set, compute their properties
        self._update_subgoal_properties(robot_pose, self.goal)
        self.old_node_dict = self.new_node_dict.copy()
        assert len(self.subgoals) == len(self.subgoal_nodes)

    def _identify_subgoal_nodes(self):
        """ Loop through subgoals and get the 'input data'
        This method also finds the representitive node for each subgoal
        on the graph and pairs their image as well
        """
        # Update the inputs for each vertex point on the graph based on the new
        # observation
        self._update_node_inputs()
        clean_data = lsp_gnn.utils.prepare_input_clean_graph(
            self.subgoals, self.vertex_points, self.edge_data,
            self.new_node_dict, self.has_updated, self.semantic_grid,
            self.wall_class, self.observation, self.robot_pose
        )
        self.vertex_points = clean_data['vertex_points']
        self.edge_data = clean_data['edge_data']
        self.new_node_dict = clean_data['new_node_dict']
        self.has_updated = clean_data['has_updated']

        return clean_data['subgoal_nodes']

    def _update_node_inputs(self):
        ''' This method computes and assigns input for each of the nodes
        present on the graph and maintains a dictionary for it.
        '''
        self.new_node_dict = {}
        self.has_updated = []

        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            # If the vertex point exists in previous step then perform the
            # following steps ->
            if vertex_point in self.old_node_dict.keys():
                input_data = self.old_node_dict[vertex_point]
                self.has_updated.append(0)
            # -> Otherwise calculate input data for new vertex point
            else:
                input_data = lsp_gnn.utils.get_input_data(
                    self.semantic_grid, self.wall_class, vertex_point,
                    self.observation, self.robot_pose)
                # input_data['last_observed_pose'] = self.robot_pose
                self.has_updated.append(1)
            self.new_node_dict[vertex_point] = input_data

    def _compute_combined_data(self):
        """ This method produces a datum for the GCN and returns it.
        make_graph(datum) needs to be called prior to forwording to
        the network
        """
        is_subgoal = []
        history = []
        # image = []
        # seg_image = []
        input_vector = []
        goal_distance_vector = []

        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            # image.append(self.new_node_dict[vertex_point]['image'])
            x = self.new_node_dict[vertex_point]['x']
            y = self.new_node_dict[vertex_point]['y']
            goal_distance_vector.append(np.sqrt(
                (x - self.goal.x) ** 2 +
                (y - self.goal.y) ** 2))
            input_vector.append(self.new_node_dict[vertex_point]['input'])
            # seg_image.append(self.new_node_dict[vertex_point]['seg_image'])
            if vertex_point in self.subgoal_nodes.keys():
                is_subgoal.append(1)
                history.append(1)
            else:
                is_subgoal.append(0)
                history.append(0)
        assert len(is_subgoal) == len(self.has_updated)

        edge_features = lsp_gnn.utils.get_edge_features(
            edge_data=self.edge_data,
            vertex_points=self.vertex_points,
            node_dict=self.new_node_dict
        )

        datum = {
            'wall_class': input_vector,
            # 'image': image,
            # 'seg_image': seg_image,
            'goal_distance': goal_distance_vector,
            'is_subgoal': is_subgoal,
            'history': history,
            'edge_data': self.edge_data,
            'has_updated': self.has_updated,
            'edge_features': edge_features
        }
        return datum

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        raise NotImplementedError("Method for abstract class")


class ConditionalKnownSubgoalPlanner(ConditionalSubgoalPlanner):
    ''' This planner class is used for data generation using known map '''
    def __init__(self, goal, args, known_map, device=None,
                 semantic_grid=None, wall_class=None, do_compute_weightings=True):
        super(ConditionalKnownSubgoalPlanner, self). \
            __init__(goal, args, device, semantic_grid, wall_class)

        self.known_map = known_map
        self.inflated_known_grid = gridmap.utils.inflate_grid(
            known_map, inflation_radius=self.inflation_radius)
        _, self.get_path = gridmap.planning.compute_cost_grid_from_position(
            self.inflated_known_grid, [goal.x, goal.y])
        self.counter = 0
        self.last_saved_training_data = None
        self.do_compute_weightings = do_compute_weightings

    def _update_subgoal_properties(self, robot_pose, goal_pose):
        new_subgoals = [s for s in self.subgoals if not s.props_set]
        lsp.core.update_frontiers_properties_known(
            self.inflated_known_grid,
            self.inflated_grid,
            self.subgoals, new_subgoals,
            robot_pose, goal_pose,
            self.downsample_factor)

        if self.do_compute_weightings:
            lsp.core.update_frontiers_weights_known(self.inflated_known_grid,
                                                    self.inflated_grid,
                                                    self.subgoals, new_subgoals,
                                                    robot_pose, goal_pose,
                                                    self.downsample_factor)

    def compute_training_data(self):
        """ This method produces training datum for both AutoEncoder and
        GCN model.
        """
        prob_feasible = []
        delta_success_cost = []
        exploration_cost = []
        positive_weighting_vector = []
        negative_weighting_vector = []
        data = self._compute_combined_data()
        for idx, node in enumerate(self.vertex_points):
            p = tuple(node)
            if p in self.subgoal_nodes.keys():
                if self.subgoal_nodes[p].prob_feasible == 1.0 \
                   and data['has_updated'][idx] != 1:
                    # Checks if an alternate subgoal path is still reachable.
                    # This changes the label for the subgoal whose
                    # alternate path has already explored the merging
                    # point through this subgoal
                    is_reachable = lsp_gnn.utils.check_if_reachable(
                        self.inflated_known_grid, self.inflated_grid,
                        self.goal, self.robot_pose, self.subgoal_nodes[p])
                    if is_reachable is False:
                        data['has_updated'][idx] = 1
                prob_feasible.append(self.subgoal_nodes[p].prob_feasible)
                delta_success_cost.append(self.subgoal_nodes[p].delta_success_cost)
                exploration_cost.append(self.subgoal_nodes[p].exploration_cost)
                positive_weighting_vector.append(self.subgoal_nodes[p].positive_weighting)
                negative_weighting_vector.append(self.subgoal_nodes[p].negative_weighting)
            else:
                prob_feasible.append(0)
                delta_success_cost.append(0)
                exploration_cost.append(0)
                positive_weighting_vector.append(0)
                negative_weighting_vector.append(0)

        data['is_feasible'] = prob_feasible
        data['delta_success_cost'] = delta_success_cost
        data['exploration_cost'] = exploration_cost
        data['positive_weighting'] = positive_weighting_vector
        data['negative_weighting'] = negative_weighting_vector
        data['known_map'] = self.known_map
        data['observed_map'] = self.observed_map
        data['subgoals'] = self.original_subgoal
        data['vertex_points'] = self.vertex_points
        data['goal'] = self.goal
        data['semantic_grid'] = self.semantic_grid
        data['semantic_labels'] = self.wall_class

        return data

    def save_training_data(self, training_data, prob=.2):
        do_save = np.random.choice([True, False], 1, p=[prob, 1 - prob])
        if do_save:
            # Before saving check so that it is not identical to last saved data
            if self.last_saved_training_data:
                if lsp_gnn.utils.check_if_same(
                        training_data, self.last_saved_training_data):
                    return
            lsp_gnn.utils.write_datum_to_file(self.args,
                                              training_data,
                                              self.counter)
            self.last_saved_training_data = training_data
            self.counter += 1

    def compute_selected_subgoal(self):
        """Use the known map to compute the selected subgoal."""
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose,
                                                  self.goal, self.subgoals)
        if is_goal_in_range:
            print("Goal in Range")
            return None
        if not self.subgoals:
            return None

        # Compute the plan
        did_plan, path = self.get_path([self.robot_pose.x, self.robot_pose.y],
                                       do_sparsify=False,
                                       do_flip=True,
                                       bound=None)
        if did_plan is False:
            print("Plan did not succeed...")
            raise NotImplementedError("Not sure what to do here yet")
        if np.argmax(self.observed_map[path[0, -1], path[1, -1]] >= 0):
            return None

        # Determine the chosen subgoal
        ind = np.argmax(self.observed_map[path[0, :], path[1, :]] < 0)
        return min(self.subgoals,
                   key=lambda s: s.get_distance_to_point((path.T)[ind]))


class ConditionalUnknownSubgoalPlanner(ConditionalSubgoalPlanner):
    ''' This planner class is used for planning under uncertainty by estimating the
    properties of the subgoals using a trained graph neural network '''
    def __init__(self, goal, args, semantic_grid=None,
                 wall_class=None, device=None):
        super(ConditionalUnknownSubgoalPlanner, self). \
            __init__(goal, args, device, semantic_grid, wall_class)

        self.out = None

        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        self.super_node_latent_features = None
        # if args.input_type == 'image' or self.args.input_type == 'seg_image':
        #     self.latent_features_net = AutoEncoder. \
        #         get_net_eval_fn_old(
        #             args.autoencoder_network_file, device=self.device,
        #             preprocess_for='Cond_Eval', args=self.args)
        #     self.subgoal_property_net = LSPConditionalGNN.get_net_eval_fn(
        #         args.network_file, device=self.device)
        if args.input_type == 'wall_class' and args.train_marginal_lsp:
            self.subgoal_property_net = WallClassGNN.get_net_eval_fn(
                args.network_file, device=self.device)
        # elif args.use_clip:
        #     self.latent_features_net = AutoEncoder.get_net_eval_fn(
        #         args.clip_network_file, device=self.device,
        #         preprocess_for='Cond_Eval')

    def _compute_cnn_data(self):
        """ This method produces datum for the AutoEncoder """
        images = []
        is_subgoal = []

        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            # Check if the latent features need to be recomputed for the vertex
            # points
            if 'latent_features' not in self.new_node_dict[vertex_point].keys():
                images.append(self.new_node_dict[vertex_point]['image'])
                if vertex_point in self.subgoal_nodes.keys():
                    is_subgoal.append(1)
                else:
                    is_subgoal.append(0)

        if self.super_node_latent_features is None:
            images.append(np.zeros((128, 512, 3)))
            is_subgoal.append(0)
        datum = {
            'image': images,
            'is_subgoal': is_subgoal,
        }
        return datum

    def _calculate_latent_features(self):
        ''' This method computes and assigns latent features to their
        respective node.
        '''
        self.cnn_input = self._compute_cnn_data()
        if self.cnn_input['image']:  # Check if cnn_input is not empty
            latent_features = self.latent_features_net(
                datum=self.cnn_input)

        ii = 0
        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            # Checks and assigns latent features to their nodes
            if 'latent_features' not in self.new_node_dict[vertex_point].keys():
                self.new_node_dict[vertex_point]['latent_features'] = \
                    latent_features[ii][None, :]    # expanding one dimension
                ii += 1
        if self.super_node_latent_features is None:
            self.super_node_latent_features = latent_features[-1][None, :]

    def _compute_gcn_data(self):
        """ This method produces a datum for the GCN and returns it.
        make_graph(datum) needs to be called prior to forwording to
        the network
        """
        # Prior to running GCN, CNN must create the latent features
        if self.args.input_type == 'image' or \
                self.args.input_type == 'seg_image':
            self._calculate_latent_features()
            latent_features = torch.zeros(0).to(self.device)
        elif self.args.input_type == 'wall_class':
            latent_features = []
            distance_features = []
        is_subgoal = []
        history = []

        for vertex_point in self.vertex_points:
            vertex_point = tuple(vertex_point)
            if self.args.input_type == 'image' or \
                    self.args.input_type == 'seg_image':
                latent_features = torch.cat((
                    latent_features,
                    self.new_node_dict[vertex_point]['latent_features']), 0)
            elif self.args.input_type == 'wall_class':
                x = self.new_node_dict[vertex_point]['x']
                y = self.new_node_dict[vertex_point]['y']
                distance_features.append(np.sqrt(
                    (x - self.goal.x) ** 2 +
                    (y - self.goal.y) ** 2))
                latent_features.append(
                    self.new_node_dict[vertex_point]['input'])
            if vertex_point in self.subgoal_nodes.keys():
                # taking note of the node being a subgoal
                is_subgoal.append(1)
                # marking if the subgoal will participate
                # in conditional probability
                history.append(1)
            else:
                is_subgoal.append(0)
                history.append(0)

        edge_features = lsp_gnn.utils.get_edge_features(
            edge_data=self.edge_data,
            vertex_points=self.vertex_points,
            node_dict=self.new_node_dict
        )

        # Add the super node
        super_node_idx = len(history)
        if self.args.input_type == 'image' or \
                self.args.input_type == 'seg_image':
            latent_features = torch.cat(
                (latent_features, self.super_node_latent_features), 0)
        elif self.args.input_type == 'wall_class':
            latent_features.append([0, 0, 0])
            latent_features = torch.tensor(latent_features, dtype=torch.float)

        is_subgoal.append(0)
        history = [0] * len(history)
        history.append(1)
        old_edges = [edge_pair for edge_pair in self.edge_data]
        new_edges = [(idx, super_node_idx) for idx in range(super_node_idx)]
        updated_edges = old_edges + new_edges

        # Add feature for each new edges connected to the super node
        for distance in distance_features:
            feature_vector = []
            feature_vector.append(distance)
            edge_features.append(feature_vector)
        distance_features.append(0)
        datum = {
            'is_subgoal': is_subgoal,
            'history': history,
            'edge_data': updated_edges,
            'edge_features': edge_features,
            'latent_features': latent_features,
            'goal_distance': distance_features,
        }
        return datum

    def _update_subgoal_properties(self,
                                   robot_pose,
                                   goal_pose):
        self.gcn_graph_input = self._compute_gcn_data()
        prob_feasible_dict, dsc, ec, out = self.subgoal_property_net(
            datum=self.gcn_graph_input,
            vertex_points=self.vertex_points,
            subgoals=self.subgoals
        )
        for subgoal in self.subgoals:
            subgoal.set_props(
                prob_feasible=prob_feasible_dict[subgoal],
                delta_success_cost=dsc[subgoal],
                exploration_cost=ec[subgoal],
                last_observed_pose=robot_pose)
            if self.verbose:
                print(
                    f'Ps={subgoal.prob_feasible:8.4f}|'
                    f'Rs={subgoal.delta_success_cost:8.4f}|'
                    f'Re={subgoal.exploration_cost:8.4f}'
                )

        if self.verbose:
            print(" ")

        self.out = out
        # The line below is not getting updated to the last elimination pass
        # that is happening before cost calculation, however this is not that
        # important because the self.is_subgoal is only used to plot the graph
        # not in cost calculation. Will fix it if necessary.
        self.is_subgoal = self.gcn_graph_input['is_subgoal'].copy()

    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose,
                                                  self.goal, self.subgoals)
        if is_goal_in_range:
            print("Goal in Range")
            return None
        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp_gnn.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                self.vertex_points.copy(),
                self.subgoal_nodes.copy(),
                self.gcn_graph_input.copy(),
                self.subgoal_property_net,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            raise ValueError()
            print("Problem with planning.")
            return None
        self.latest_ordering = frontier_ordering
        self.selected_subgoal = list(self.subgoals)[frontier_ordering[0]]
        return self.selected_subgoal


class LSP(ConditionalUnknownSubgoalPlanner):
    def __init__(self, goal, args, device=None, verbose=False,
                 semantic_grid=None, wall_class=None):
        super(LSP, self).__init__(goal, args, device, semantic_grid, wall_class)

        if device is not None:
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        # if args.input_type == 'image' or args.input_type == 'seg_image':
        #     self.latent_features_net = AutoEncoder.get_net_eval_fn_old(
        #         args.autoencoder_network_file, device=self.device,
        #         preprocess_for='CNN_Eval', args=self.args)
        #     self.subgoal_property_net = CNNLSP.get_net_eval_fn(
        #         args.cnn_network_file, device=self.device)
        if args.input_type == 'wall_class':
            self.subgoal_property_net = WallClassLSP.get_net_eval_fn(
                args.cnn_network_file, device=self.device)

        self.semantic_grid = semantic_grid
        self.wall_class = wall_class

    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose, self.goal,
                                                  self.subgoals)

        if is_goal_in_range:
            print("Goal in Range")
            return None

        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor,
                do_correct_low_prob=True))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            print("Problem with planning.")
            self.latest_ordering = None
            self.selected_subgoal = None
            return None

        self.latest_ordering = frontier_ordering
        self.selected_subgoal = frontier_ordering[0]
        return self.selected_subgoal


class GCNLSP(ConditionalUnknownSubgoalPlanner):
    def __init__(self, goal, args, semantic_grid=None,
                 wall_class=None, device=None):
        args.network_file = args.gcn_network_file
        super(GCNLSP, self).__init__(goal, args, semantic_grid, wall_class, device)
        if args.input_type == 'wall_class':
            self.subgoal_property_net = WallClassGNN.get_net_eval_fn(
                args.network_file, device=self.device)

    def compute_selected_subgoal(self):
        is_goal_in_range = lsp.core.goal_in_range(self.inflated_grid,
                                                  self.robot_pose, self.goal,
                                                  self.subgoals)

        if is_goal_in_range:
            print("Goal in Range")
            return None

        # Compute chosen frontier
        min_cost, frontier_ordering = (
            lsp.core.get_best_expected_cost_and_frontier_list(
                self.inflated_grid,
                self.robot_pose,
                self.goal,
                self.subgoals,
                num_frontiers_max=NUM_MAX_FRONTIERS,
                downsample_factor=self.downsample_factor,
                do_correct_low_prob=True))
        if min_cost is None or min_cost > 1e8 or frontier_ordering is None:
            print("Problem with planning.")
            self.latest_ordering = None
            self.selected_subgoal = None
            return None

        self.latest_ordering = frontier_ordering
        self.selected_subgoal = frontier_ordering[0]
        return self.selected_subgoal
