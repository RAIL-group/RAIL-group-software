import learning
import tempfile
import numpy as np
import os


def test_learning_data_compressed_save_load():
    """Show that we can save and load a dictionary."""

    with tempfile.TemporaryDirectory() as pickle_dir:
        pickle_filename = os.path.join(pickle_dir, 'a_pickle.pickle.gz')

        datum = {'something': 'a string',
                 'another': 104.3,
                 'numpy': np.random.rand(5, 5)}
        print("Writing data:")
        print(datum)

        learning.data.write_compressed_pickle(pickle_filename, datum)
        datum_loaded = learning.data.load_compressed_pickle(pickle_filename)

        # Some print statements
        print("Loaded data:")
        print(datum_loaded)

        assert datum['something'] == datum_loaded['something']
        assert datum['another'] == datum_loaded['another']
        assert (datum['numpy'] == datum_loaded['numpy']).all()
