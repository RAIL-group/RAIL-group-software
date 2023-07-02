def generate_all_rollout_history(history):
    '''TODO - This funtion will overflow memory at some point, need to think
    of an alternative
    '''
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


def image_aligned_to_non_subgoal(image, r_pose, vertex_point):
    """Permutes an image from axis-aligned to subgoal-pointing frame.
    The subgoal should appear at the center of the image."""
    cols = image.shape[1]
    sp = vertex_point
    yaw = np.arctan2(sp[1] - r_pose.y, sp[0] - r_pose.x) - r_pose.yaw
    roll_amount = int(round(-cols * yaw / (2 * math.pi)))
    return np.roll(image, shift=roll_amount, axis=1)


def get_rel_goal_loc_vecs(pose, goal_pose, num_bearing, vertex_point=None):
    # Lookup vectors
    _, vec_bearing = lsp.utils.learning_vision.get_directions(num_bearing)
    if vertex_point is None:
        vec_bearing = vec_bearing + pose.yaw
    else:
        sp = vertex_point
        vertex_point_yaw = np.arctan2(sp[1] - pose.y, sp[0] - pose.x)
        vec_bearing = vec_bearing + vertex_point_yaw

    robot_point = np.array([pose.x, pose.y])
    goal_point = np.array([goal_pose.x, goal_pose.y])
    rel_goal_point = goal_point - robot_point

    goal_loc_x_vec = rel_goal_point[0] * np.cos(
        vec_bearing) + rel_goal_point[1] * np.sin(vec_bearing)
    goal_loc_y_vec = -rel_goal_point[0] * np.sin(
        vec_bearing) + rel_goal_point[1] * np.cos(vec_bearing)

    return (goal_loc_x_vec[:, np.newaxis].T, goal_loc_y_vec[:, np.newaxis].T)


def get_path_middle_point(known_map, start, goal, args):
    """This function returns the middle point on the path from goal to the
    robot starting position"""
    inflation_radius = args.inflation_radius_m / args.base_resolution
    inflated_mask = gridmap.utils.inflate_grid(known_map,
                                               inflation_radius=inflation_radius)
    # Now sample the middle point
    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        inflated_mask, [goal.x, goal.y])
    _, path = get_path([start.x, start.y],
                       do_sparsify=False,
                       do_flip=False)
    row, col = path.shape
    x = path[0][col // 2]
    y = path[1][col // 2]
    new_start_pose = common.Pose(x=x,
                                 y=y,
                                 yaw=2 * np.pi * np.random.rand())
    return new_start_pose


def get_wall_color_vector(seg_img, robot_pose=None,
                          vertex_point=None, cone=None):
    '''This method gets the [red, blue, gray]
    '''

    # Re-orient the image based on the subgoal centroid
    if robot_pose is not None and vertex_point is not None:
        seg_img = image_aligned_to_non_subgoal(
            seg_img, robot_pose, vertex_point)
    seg_img = np.transpose(seg_img, (2, 0, 1))
    if cone == 180:
        seg_img = seg_img[:, :, 128:384]
    elif cone == 90:
        seg_img = seg_img[:, :, 128 + 64:384 - 64]
    elif cone == 45:
        seg_img = seg_img[:, :, 128 + 64 + 32:384 - 64 - 32]
    # plt.imshow(np.transpose(seg_img, (1, 2, 0)))
    # plt.show()
    blue_count = np.count_nonzero(seg_img[2] == 255)
    red_count = np.count_nonzero(seg_img[0] == 255)
    gray_count = np.count_nonzero(seg_img[1] == 127)
    # brown_count = np.count_nonzero(seg_img[1] == 63)
    # green_count = np.count_nonzero(seg_img[1] == 255)
    poi_count = red_count + blue_count + gray_count
    vect = [red_count / poi_count,
            blue_count / poi_count,
            gray_count / poi_count]
    return np.array(vect)


def preprocess_cnn_training_data(fn, args=None):
    ''' This method preprocesses the data for training base/cnn lsp
    We are interested in only training the subgoal images because that is what
    the actual implementation did as well
    '''
    def preprocess(data):
        datum = {}
        features = [
            'is_feasible', 'delta_success_cost', 'exploration_cost',
            'positive_weighting', 'negative_weighting'
        ]
        if fn is None:
            ds_img_list = []
            for idx, raw_image in enumerate(data['image']):
                if data['is_subgoal'][idx] == 1:
                    down_sampled_img = downsample_image(
                        np.transpose(raw_image, (2, 0, 1)))
                    # take the middle 1/8 of the image:
                    down_sampled_img = down_sampled_img[:, :, 56:72]
                    ds_img_list.append(down_sampled_img)
            datum['image'] = torch.tensor(
                (np.array(ds_img_list).astype(np.float32) / 255), dtype=torch.float)
        else:  # Do the preprocessing for running clip encoder
            datum['image'] = torch.cat([
                fn(Image.fromarray(image, mode='RGB')).unsqueeze(0)
                for idx, image in enumerate(data['image'])
                if data['is_subgoal'][idx] == 1
            ])
        for feature in features:
            datum[feature] = torch.tensor(np.array([
                feature_value
                for idx, feature_value in enumerate(data[feature])
                if data['is_subgoal'][idx] == 1
            ]))

        return Data(x=datum['image'],
                    y=datum['is_feasible'],
                    dsc=datum['delta_success_cost'],
                    ec=datum['exploration_cost'],
                    pweight=datum['positive_weighting'],
                    nweight=datum['negative_weighting'])

    def no_ae_preprocess(data):
        datum = {}
        features = [
            'is_feasible', 'delta_success_cost', 'exploration_cost',
            'positive_weighting', 'negative_weighting'
        ]
        datum['wall_class'] = torch.tensor(
            [vector
             for idx, vector in enumerate(data['wall_class'])
             if data['is_subgoal'][idx] == 1],
            dtype=torch.float)
        for feature in features:
            datum[feature] = torch.tensor(np.array([
                feature_value
                for idx, feature_value in enumerate(data[feature])
                if data['is_subgoal'][idx] == 1
            ]))

        return Data(x=datum['wall_class'],
                    y=datum['is_feasible'],
                    dsc=datum['delta_success_cost'],
                    ec=datum['exploration_cost'],
                    pweight=datum['positive_weighting'],
                    nweight=datum['negative_weighting'])

    def seg_image_preprocess(data):
        datum = {}
        features = [
            'is_feasible', 'delta_success_cost', 'exploration_cost',
            'positive_weighting', 'negative_weighting'
        ]
        ds_img_list = []

        for idx, raw_image in enumerate(data['seg_image']):
            if data['is_subgoal'][idx] == 1:
                down_sampled_img = np.transpose(
                    raw_image, (2, 0, 1))[:, ::4, ::4]
                # take the middle 1/8 of the image:
                image = down_sampled_img[:, :, 56:72]
                image = convert_seg_image_to_one_hot_coded_vector(image)
                ds_img_list.append(image)
        datum['image'] = torch.tensor(
            (np.array(ds_img_list).astype(np.float32) / 255), dtype=torch.float)

        for feature in features:
            datum[feature] = torch.tensor(np.array([
                feature_value
                for idx, feature_value in enumerate(data[feature])
                if data['is_subgoal'][idx] == 1
            ]))

        return Data(x=datum['image'],
                    y=datum['is_feasible'],
                    dsc=datum['delta_success_cost'],
                    ec=datum['exploration_cost'],
                    pweight=datum['positive_weighting'],
                    nweight=datum['negative_weighting'])

    if args.input_type == 'image':
        return preprocess
    elif args.input_type == 'seg_image':
        return seg_image_preprocess
    elif args.input_type == 'wall_class':
        return no_ae_preprocess