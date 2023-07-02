import os.path
import pytest


def pytest_addoption(parser):
    """Shared command line options for pytest."""
    parser.addoption("--xpassthrough", default="false", action="store")
    parser.addoption("--unity-path", default=None, action="store")
    parser.addoption("--lsp-xai-maze-net-0SG-path", default=None, action="store")
    parser.addoption("--lsp-select-network-file", default=None, action="store")
    parser.addoption("--lsp-select-generator-network-file", default=None, action="store")


@pytest.fixture()
def do_debug_plot(pytestconfig):
    if pytestconfig.getoption("xpassthrough") == 'true':
        return True

    return False


@pytest.fixture()
def unity_path(pytestconfig):
    unity_path_str = pytestconfig.getoption("unity_path")
    if unity_path_str is None:
        pytest.skip("Unity path not provided")

    if not os.path.exists(unity_path_str):
        raise ValueError(f"Unity path '{unity_path_str}' does not exist.")

    return unity_path_str
