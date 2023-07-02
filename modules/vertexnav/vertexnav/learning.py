import numpy as np
import vertexnav


def get_vertex_datum_for_pose(pose,
                              world,
                              unity_bridge,
                              max_range,
                              num_range,
                              num_bearing,
                              min_range=None,
                              pano_image=None):
    """Get all relevant data for a single pose."""
    if unity_bridge is not None:
        unity_bridge.move_object_to_pose("robot", pose)
        pano_image = unity_bridge.get_image("robot/pano_camera")
        pano_image = vertexnav.utils.convert.image_aligned_to_robot(
            pano_image, pose) / 255
    pishape = pano_image.shape

    # Reshape the image if necessary and error handling
    if 2 * pishape[0] == pishape[1]:
        pano_image = pano_image[pishape[0] // 4:3 * pishape[0] // 4]
    elif not 4 * pishape[0] == pishape[1]:
        raise ValueError(f"Image aspect ratio unsupported (shape {pishape})")

    pano_depth_image = unity_bridge.get_image("robot/pano_depth_camera")
    pano_depth_image = vertexnav.utils.convert.depths_from_depth_image(
        pano_depth_image)
    sh = pano_depth_image.shape
    minimum_range = min(pano_depth_image[sh[0] // 2])
    if min_range is not None and minimum_range < min_range:
        print("FAILED: within minimum range.")
        return None

    pano_depth_image = pano_depth_image[sh[0] // 4:3 * sh[0] // 4]
    pano_depth_image = vertexnav.utils.convert.image_aligned_to_robot(
        pano_depth_image, pose)

    perfect_gap_obs = vertexnav.noisy.convert_world_obs_to_noisy_detection(
        world.get_vertices_for_pose(pose, max_range=max_range),
        pose,
        do_add_noise=False)

    vertex_data = vertexnav.utils.convert.get_vertex_grid_data_from_obs(
        observation=perfect_gap_obs,
        size=pano_image.shape[1],
        pose=pose,
        max_range=max_range,
        num_range=num_range,
        num_bearing=num_bearing)

    return {
        'pose_x': pose.x,
        'pose_y': pose.y,
        'pose_yaw_rad': pose.yaw,
        'image': pano_image,
        'depth': pano_depth_image.astype(np.float32),
        'image_size': pano_image.shape[:2],
        'output_size': vertex_data['is_vertex'].shape,
        'is_vertex': vertex_data['is_vertex'],
        'is_left_gap': vertex_data['is_left_gap'],
        'is_right_gap': vertex_data['is_right_gap'],
        'is_corner': vertex_data['is_corner'],
        'is_point_vertex': vertex_data['is_point_vertex'],
        'is_in_view': vertex_data['is_in_view'],
    }
