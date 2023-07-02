class InfoKnownSubgoalPlanner(ConditionalKnownSubgoalPlanner):
    def __init__(self, goal, args, known_map, semantic_grid=None,
                 wall_class=None, device=None, do_compute_weightings=True):
        super(InfoKnownSubgoalPlanner, self). \
            __init__(goal, args, known_map, device, semantic_grid, wall_class)
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device
        if args.input_type == 'wall_class':
            self.subgoal_property_net = WallClassGNN.get_net_eval_fn(
                args.gcn_network_file, device=self.device)

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
                    is_reachable = lsp_cond.utils.check_if_reachable(
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
        
        value_of_information_vector = lsp_cond.utils.get_value_of_information_vector(
            self.subgoal_nodes.copy(), self.vertex_points, self.edge_data,
            self.semantic_grid, self.wall_class, self.robot_pose, self.goal,
            self.subgoal_property_net, self.downsample_factor,
            self.args.inflation_radius_m, self.args.base_resolution,
            self.inflated_grid, self.observed_map, self.known_map
        )
        assert len(value_of_information_vector) == len(prob_feasible)

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
        data['value_of_information'] = value_of_information_vector

        return data
