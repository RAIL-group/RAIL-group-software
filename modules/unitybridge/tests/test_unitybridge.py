import numpy as np
import pytest
import unitybridge


@pytest.mark.timeout(5)
def test_unitybridge_loads_without_crashing(unity_path):
    """The UnityBridge class can load the env without crashing or hanging."""
    with unitybridge.UnityBridge(unity_path) as _:
        pass


@pytest.mark.timeout(15)
def test_unitybridge_get_image(unity_path, do_debug_plot):
    """The unity_bridge can be used to generate images."""
    with unitybridge.UnityBridge(unity_path) as unity_bridge:
        pano_image = unity_bridge.get_image(
            "robot/pano_camera")
        pano_depth_image = unity_bridge.get_image(
            "robot/pano_depth_camera")
        pano_segmentation_image = unity_bridge.get_image(
            "robot/pano_segmentation_camera")

        if do_debug_plot:
            import matplotlib.pyplot as plt
            plt.subplot(131)
            plt.imshow(pano_image)
            plt.title("Pano Image")

            plt.subplot(132)
            plt.imshow(pano_depth_image)
            plt.title("Pano Depth Image")

            plt.subplot(133)
            plt.imshow(pano_segmentation_image)
            plt.title("Pano Segmentation Image")

            plt.show()

        assert pano_image.size > 0
        assert np.std(pano_image) > 0
        assert pano_depth_image.size > 0
        assert np.std(pano_depth_image) > 0
        assert pano_segmentation_image.size > 0
        assert np.std(pano_segmentation_image) > 0


@pytest.mark.timeout(15)
def test_unitybridge_can_add_objects(unity_path, do_debug_plot):
    """Rendered images should change after a few objects are added."""
    with unitybridge.UnityBridge(unity_path) as unity_bridge:
        pano_image = unity_bridge.get_image(
            "robot/pano_camera")
        pano_depth_image = unity_bridge.get_image(
            "robot/pano_depth_camera")

        for _ in range(10):
            unity_bridge.create_cube()

        pano_image_cubes = unity_bridge.get_image(
            "robot/pano_camera")
        pano_depth_image_cubes = unity_bridge.get_image(
            "robot/pano_depth_camera")

        if do_debug_plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(pano_image_cubes)
            plt.title("Pano Image")

            plt.subplot(122)
            plt.imshow(pano_depth_image_cubes)
            plt.title("Pano Depth Image")

            plt.show()

        assert np.std(pano_image - pano_image_cubes) > 0
        assert np.std(pano_depth_image - pano_depth_image_cubes) > 0
