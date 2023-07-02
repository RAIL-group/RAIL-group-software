import vertexnav


def follow_path_iterator(path, segment_steps=50, smooth_factor=1.0 / 3):
    """Loop through a pose and return a smoothed path (of Poses)."""
    # First get the smoothed path
    points = vertexnav.utils.calc.smooth_path(path,
                                              segment_steps=segment_steps,
                                              smooth_factor=smooth_factor)

    # Convert to poses
    poses = [vertexnav.Pose(point[0], point[1]) for point in points]

    for pose in poses:
        yield pose
