import vertexnav
import matplotlib.pyplot as plt
import pytest


def test_vertexnav_can_generate_dungeon(do_debug_plot):
    world = vertexnav.environments.dungeon.DungeonWorld(random_seed=13)

    if do_debug_plot:
        vertexnav.plotting.plot_world(plt.gca(), world)
        plt.show()


def test_vertexnav_dungeon_seed_control(do_debug_plot):
    """Confirm that setting the seed controls the generation."""
    seed_a, seed_b, seed_c = 13, 10, 13
    world_a = vertexnav.environments.dungeon.DungeonWorld(random_seed=seed_a)
    world_b = vertexnav.environments.dungeon.DungeonWorld(random_seed=seed_b)
    world_c = vertexnav.environments.dungeon.DungeonWorld(random_seed=seed_c)
    ksp_a = world_a.known_space_poly
    ksp_b = world_b.known_space_poly
    ksp_c = world_c.known_space_poly

    if do_debug_plot:
        plt.subplot(131)
        vertexnav.plotting.plot_world(plt.gca(), world_a)
        plt.title(f"Seed: {seed_a}")
        plt.subplot(132)
        vertexnav.plotting.plot_world(plt.gca(), world_b)
        plt.title(f"Seed: {seed_b}")
        plt.subplot(133)
        vertexnav.plotting.plot_world(plt.gca(), world_c)
        plt.title(f"Seed: {seed_c}")
        plt.show()

    assert not seed_a == seed_b
    assert ksp_a.intersection(ksp_b).area / ksp_a.area < 1.0
    assert seed_a == seed_c
    assert ksp_a.intersection(ksp_c).area / ksp_a.area == pytest.approx(1.0)
