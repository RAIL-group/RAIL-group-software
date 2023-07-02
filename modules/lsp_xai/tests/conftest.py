import os.path
import pytest


def pytest_addoption(parser):
    """Shared command line options for pytest."""
    pass


@pytest.fixture()
def lsp_xai_maze_net_0SG_path(pytestconfig):
    net_path_str = pytestconfig.getoption("lsp_xai_maze_net_0SG_path")
    if net_path_str is None:
        pytest.skip("Net path not provided")

    if not os.path.exists(net_path_str):
        raise ValueError("Net path '{net_path_str}' does not exist.")

    return net_path_str
