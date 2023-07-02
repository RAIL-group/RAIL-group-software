import os.path
import pytest


def pytest_addoption(parser):
    """Shared command line options for pytest."""
    pass


@pytest.fixture()
def lsp_select_network_file(pytestconfig):
    net_path_str = pytestconfig.getoption("--lsp-select-network-file")
    if net_path_str is not None:
        if not os.path.exists(net_path_str):
            raise ValueError(f"Net path '{net_path_str}' does not exist.")
    return net_path_str


@pytest.fixture()
def lsp_select_generator_network_file(pytestconfig):
    net_path_str = pytestconfig.getoption("--lsp-select-generator-network-file")
    if net_path_str is not None:
        if not os.path.exists(net_path_str):
            raise ValueError(f"Net path '{net_path_str}' does not exist.")
    return net_path_str
