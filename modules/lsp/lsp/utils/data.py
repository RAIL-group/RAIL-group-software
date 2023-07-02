import gzip
import _pickle as cPickle
import os

COMPRESS_LEVEL = 2


def write_training_data_to_pickle(data, step_counter, args):
    # Get the data from the planner
    # Ensure data is a list
    try:
        iter(data)
    except TypeError:
        data = [data]

    csv_full_path = get_csv_filename(args)

    pickle_names = []
    for ii, datum in enumerate(data):
        data_filename = os.path.join(
            'data',
            f'dat_{args.current_seed}_{step_counter}_{ii}.pgz'
        )

        pickle_name = os.path.join(args.save_dir, data_filename)
        with gzip.GzipFile(pickle_name, 'wb',
                           compresslevel=COMPRESS_LEVEL) as f:
            cPickle.dump(datum, f, protocol=-1)

        with open(csv_full_path, 'a') as f:
            f.write(f'{data_filename}\n')

        pickle_names.append(pickle_name)

    return pickle_names


def get_csv_filename(args):
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.csv'
    return os.path.join(args.save_dir, csv_filename)
