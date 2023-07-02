import example
import pytest


@pytest.mark.parametrize("mat_shape", [(1, 4), (2, 2), (2,), (1, 2, 3, 4)])
def test_get_random_matrix_is_correct_size(mat_shape):
    mat = example.core.get_random_matrix(mat_shape)
    assert mat.shape == mat_shape
    assert mat.min() >= 0.0
    assert mat.max() <= 1.0


def test_optional_demo_plot(do_debug_plot):
    """Show that if plotting is enabled, the plot will show during testing. Note
that the optional argument inherits from 'XPASSTHROUGH' and is set in the
'fixtures' defined in the top-level testing directory."""
    if do_debug_plot:
        import matplotlib.pyplot as plt
        mat = example.core.get_random_matrix((25, 25))
        plt.figure(figsize=(8, 8))
        plt.imshow(mat)
        plt.title("Debug: we can plot from inside a test")
        plt.show()
