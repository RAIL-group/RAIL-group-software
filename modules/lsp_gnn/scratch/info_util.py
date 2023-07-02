import gridmap
import lsp_cond
from skimage.morphology import dilation


def get_frontier_revealing_grid(grid_map, known_map, frontier, frontiers):
    ''' This function adds the region through the given frontier to the currently observed
    grid. So that we can generate graph on it.
    '''
    f_remove_idx = []
    free = known_map != COLLISION_VAL
    unknown = grid_map == UNOBSERVED_VAL
    unknown_free = np.logical_and(free, unknown)

    grid = np.ones([grid_map.shape[0], grid_map.shape[1]])
    grid[unknown_free] = FREE_VAL
    labeled_con_comp = skimage.measure.label(grid, background=1)
    f_points = frontier.get_frontier_point()
    if labeled_con_comp[f_points[0], f_points[1]] == 0:  # if same as background nothing to reveal
        result = grid_map.copy()
        # result[frontier.points] = COLLISION_VAL
        pt_x, pt_y = frontier.points[0], frontier.points[1]
        for x, y in zip(pt_x, pt_y):
            result[x, y] = COLLISION_VAL
        return result, f_remove_idx

    region_of_interest = labeled_con_comp == labeled_con_comp[f_points[0], f_points[1]]

    # Figure out which of the subgoal should be removed because they share the same
    # revealed region
    for idx, f in enumerate(frontiers):
        f_p = f.get_frontier_point()
        if labeled_con_comp[f_points[0], f_points[1]] == \
           labeled_con_comp[f_p[0], f_p[1]]:
            f_remove_idx.append(idx)

    dialated_roi = dilation(region_of_interest, footprint=lsp_cond.plotting.FOOT_PRINT)
    result = grid_map.copy()
    result[dialated_roi] = known_map[dialated_roi]
    return result, f_remove_idx


def get_inflated_occupancy_grid(observed_map, inflation_radius, robot_pose):
    """Compute the inflated grid."""
    # Inflate the grid and generate a plan
    inflated_grid = gridmap.utils.inflate_grid(
        observed_map, inflation_radius=inflation_radius)

    inflated_grid = gridmap.mapping.get_fully_connected_observed_grid(
        inflated_grid, robot_pose)
    # Prevents robot from getting stuck occasionally: sometimes (very
    # rarely) the robot would reveal an obstacle and then find itself
    # within the inflation radius of that obstacle. This should have
    # no side-effects, since the robot is expected to be in free space.
    inflated_grid[int(robot_pose.x), int(robot_pose.y)] = 0

    return inflated_grid


def update_node_inputs(vertex_points, semantic_grid, wall_class, robot_pose):
    new_node_dict = {}
    has_updated = []
    for vertex_point in vertex_points:
        vertex_point = tuple(vertex_point)
        input_data = lsp_cond.utils.get_input_data(
            semantic_grid, wall_class, vertex_point)
        has_updated.append(1)
        new_node_dict[vertex_point] = input_data
    return new_node_dict


def get_gcn_data_adding_supernode(
        vertex_points, edge_data, subgoal_nodes, new_node_dict, goal):
    latent_features = []
    distance_features = []
    is_subgoal = []
    history = []

    for vertex_point in vertex_points:
        vertex_point = tuple(vertex_point)
        x = new_node_dict[vertex_point]['x']
        y = new_node_dict[vertex_point]['y']
        distance_features.append(np.sqrt(
            (x - goal.x) ** 2 +
            (y - goal.y) ** 2))
        latent_features.append(
            new_node_dict[vertex_point]['input'])
        if vertex_point in subgoal_nodes.keys():
            is_subgoal.append(1)
            history.append(1)
        else:
            is_subgoal.append(0)
            history.append(0)

    # Add the super node
    super_node_idx = len(history)
    latent_features.append([0, 0, 0])
    latent_features = torch.tensor(latent_features, dtype=torch.float)
    distance_features.append(0)
    is_subgoal.append(0)
    history = [0] * len(history)
    history.append(1)
    old_edges = [edge_pair for edge_pair in edge_data]
    new_edges = [(idx, super_node_idx) for idx in range(super_node_idx)]
    updated_edges = old_edges + new_edges

    datum = {
        'is_subgoal': is_subgoal,
        'history': history,
        'edge_data': updated_edges,
        'latent_features': latent_features,
        'goal_distance': distance_features,
    }
    return datum


def get_subgoal_properties(vertex_points, datum, subgoals, robot_pose, subgoal_property_net):
    prob_feasible_dict, dsc, ec, out = subgoal_property_net(
        datum=datum,
        vertex_points=vertex_points,
        subgoals=subgoals
    )
    for subgoal in subgoals:
        subgoal.set_props(
            prob_feasible=prob_feasible_dict[subgoal],
            delta_success_cost=dsc[subgoal],
            exploration_cost=ec[subgoal],
            last_observed_pose=robot_pose)
    return subgoals


def get_value_of_information_vector(
        subgoal_nodes, vertex_points, edge_data,
        semantic_grid, wall_class, robot_pose, goal,
        subgoal_property_net, downsample_factor,
        inflation_radius_m, base_resolution, inflated_grid,
        observed_map, known_map):
    ''' This function calculates the value of information for all vertex-points
    that are to be saved as graph input
    '''

    # Preserve the known subgoal properties before overwriting with the estimates
    # from the neural netwokr
    for subgoal in subgoal_nodes.values():
        subgoal.known_prob_feasible = subgoal.prob_feasible
        subgoal.known_delta_success_cost = subgoal.delta_success_cost
        subgoal.known_exploration_cost = subgoal.exploration_cost

    for idx, subgoal in enumerate(list(subgoal_nodes.values())):
        node_dict = update_node_inputs(vertex_points, semantic_grid, wall_class, robot_pose)
        data = get_gcn_data_adding_supernode(
            vertex_points, edge_data, subgoal_nodes, node_dict, goal
        )
        # Getting the subgoal properties for the actual graph
        real_subgoals = get_subgoal_properties(
            vertex_points, data, list(subgoal_nodes.values()).copy(),
            robot_pose, subgoal_property_net)
        if subgoal.known_prob_feasible == 1.0:
            subgoal.value_of_information = 0
        else:
            voi = calculate_value_of_information(
                idx, real_subgoals, robot_pose, goal,
                semantic_grid, wall_class, downsample_factor,
                inflation_radius_m, base_resolution, inflated_grid,
                observed_map, known_map, subgoal_property_net)
            subgoal.value_of_information = voi

    value_of_information = []
    for idx, node in enumerate(vertex_points):
        p = tuple(node)
        if p in subgoal_nodes:
            value_of_information.append(subgoal_nodes[p].value_of_information)
        else:
            value_of_information.append(0)

    # Restore the subgoal properties to known values
    for subgoal in subgoal_nodes.values():
        subgoal.prob_feasible = subgoal.known_prob_feasible
        subgoal.delta_success_cost = subgoal.known_delta_success_cost
        subgoal.exploration_cost = subgoal.known_exploration_cost

    return value_of_information


def calculate_value_of_information(
        subgoal_idx, subgoals, robot_pose, goal,
        semantic_grid, wall_class, downsample_factor,
        inflation_radius_m, base_resolution, inflated_grid,
        observed_map, known_map, subgoal_property_net):
    # Calculate the cost and ordering excluding the subgoal that we image to be revealed
    frontiers = [f for f in subgoals]
    frontier_to_reveal = frontiers.pop(subgoal_idx)

    min_cost_before, _ = (
        lsp.core.get_best_expected_cost_and_frontier_list(
            inflated_grid,
            robot_pose,
            goal,
            frontiers,  # Here we pass the list of frontiers excluding subgoal_idx
            num_frontiers_max=lsp_cond.planners.NUM_MAX_FRONTIERS,
            downsample_factor=downsample_factor,
            do_correct_low_prob=True))

    # Get the revealed grid for the subgoal_idx to create graph
    subgoal_revealed_grid, f_remove_idx = get_frontier_revealing_grid(
        observed_map, known_map, frontier_to_reveal, frontiers
    )
    inflated_grid = get_inflated_occupancy_grid(
        subgoal_revealed_grid, inflation_radius_m / base_resolution, robot_pose
    )

    # Remove the frontiers that get revealed along with the revealed region
    for idx in f_remove_idx[::-1]:
        frontiers.pop(idx)

    uncleaned_graph = compute_skeleton(inflated_grid, frontiers)
    vertex_points = uncleaned_graph['vertex_points']
    edge_data = uncleaned_graph['edges']
    new_node_dict = {}

    clean_data = prepare_input_clean_graph(
        frontiers, vertex_points, edge_data,
        new_node_dict, [0] * len(vertex_points), semantic_grid,
        wall_class, None, robot_pose
    )

    node_dict = update_node_inputs(
        clean_data['vertex_points'], semantic_grid,
        wall_class, robot_pose)
    data = get_gcn_data_adding_supernode(
        clean_data['vertex_points'], clean_data['edge_data'],
        clean_data['subgoal_nodes'], node_dict, goal
    )
    # Getting the subgoal properties for the revealed graph
    unreal_subgoals = get_subgoal_properties(
        clean_data['vertex_points'], data, frontiers, robot_pose, subgoal_property_net)

    min_cost_after, _ = (
        lsp.core.get_best_expected_cost_and_frontier_list(
            inflated_grid,
            robot_pose,
            goal,
            unreal_subgoals,  # Here we pass the list of frontiers excluding subgoal_idx
            num_frontiers_max=lsp_cond.planners.NUM_MAX_FRONTIERS,
            downsample_factor=downsample_factor,
            do_correct_low_prob=True))

    if min_cost_after is None and min_cost_before is None:
        return 0

    return min_cost_before - min_cost_after