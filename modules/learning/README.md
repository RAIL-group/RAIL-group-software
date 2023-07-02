# Learning: Data Handling and Visualization

## Introduction

This Python module consists of two main components: a data processing component (`data.py`) and a logging component (`logging.py`). The data processing component is responsible for handling pickled files, loading and saving them with compression, and loading them into a PyTorch `Dataset` for use in machine learning applications. The logging component is designed to log data using Matplotlib for visualization via TensorBoard.

## File Overview

### data.py

- `write_compressed_pickle(pickle_filepath, datum, compresslevel=COMPRESS_LEVEL)`: This function writes a datum to a file as a compressed pickle. It compresses the pickle file using the gzip compression method with a specified compression level.
  
- `load_compressed_pickle(pickle_filepath)`: This function loads a datum from a compressed pickle file. The function uses gzip to decompress the pickle file and then loads the datum.

- `CSVPickleDataset`: This is a custom PyTorch `Dataset` class designed for data that is saved as compressed pickle files. It also optionally preprocesses the data on-the-fly during data loading.

### logging.py

- `tensorboard_plot_decorator(plot_func)`: This function is a decorator for creating Matplotlib plots to be saved and viewed in TensorBoard. The decorated function should take a `Figure` object as a keyword argument and perform its plot using that figure. The decorator takes care of creating the figure, transforming the plot into an image format suitable for TensorBoard, and writing the image to TensorBoard.

## Usage

The `tensorboard_plot_decorator` is meant to wrap a matplotlib-based plotting function so that its outputs may be stored in TensorBoard for easy visualization. It wraps a function whose signature includes the `fig` keyword, which specifies the matplotlib figure on which the plotting will be done. Here is a minimal example using this functino.

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from logging import tensorboard_plot_decorator

# Instantiate a TensorBoard writer
writer = SummaryWriter()

# Decorator applied to a plotting function
@tensorboard_plot_decorator
def plot_line(fig, x, y):
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    ax.set_title('Line Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Generate some data for plotting
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Call the decorated function
plot_line(writer, 'Line Plot', 0, x, y)  # 0 is the index (step number)

# Close the writer
writer.close()
```

