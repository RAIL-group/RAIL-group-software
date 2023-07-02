from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import numpy as np


def tensorboard_plot_decorator(plot_func):
    def wrapper(self, writer, name, index, *args, **kwds):
        fig = Figure(dpi=200)
        canvas = FigureCanvas(fig)
        kwds['fig'] = fig
        plot_func(self, *args, **kwds)
        canvas.draw()
        plot_image = np.array(canvas.renderer.buffer_rgba())[:, :, 0:3]
        plot_image = np.transpose(plot_image, (2, 0, 1))
        writer.add_image(name, plot_image, index)

    return wrapper
