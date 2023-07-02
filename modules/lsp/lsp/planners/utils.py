import os
import gzip
import _pickle as cPickle
# for Palatino and other serif fonts use:

COMPRESS_LEVEL = 2


def write_supervised_training_datum_oriented(planner):
    # Get the data from the planner
    try:
        data = planner.subgoal_data_list
        if not planner.updated_subgoals:
            return False, ''
    except AttributeError:
        raise AttributeError(
            f"Planner {type(planner).__name__} does not have 'subgoal_data_list'. "
            f"Consider using an lsp.planner.KnownSubgoalPlanner."
        )

    if not data:
        return False, []

    # Write the datum to file
    args = planner.args
    pickle_names = []
    csv_full_path = get_csv_file_supervised(args)
    for ii, datum in enumerate(data):
        data_filename = os.path.join(
            'data',
            f'dat_{args.current_seed}_{planner.update_counter}_{ii}.supervised.pgz'
        )

        pickle_name = os.path.join(args.save_dir, data_filename)
        with gzip.GzipFile(pickle_name, 'wb',
                           compresslevel=COMPRESS_LEVEL) as f:
            cPickle.dump(datum, f, protocol=-1)

        with open(csv_full_path, 'a') as f:
            f.write(f'{data_filename}\n')

        pickle_names.append(pickle_name)

    return True, pickle_names


def write_comparison_training_datum(planner, target_subgoal):
    subgoal_training_datum = None
    if target_subgoal is None:
        return False, ''

    # Get the 'target subgoal' from the chosen planner
    for s in planner.subgoals:
        if s == target_subgoal:
            target_subgoal = s
            break

    subgoal_training_datum = planner.compute_subgoal_data(target_subgoal, 24)

    if subgoal_training_datum is None:
        return False, ''

    # Write the datum to file
    args = planner.args
    csv_full_path = get_csv_file_combined(args)
    data_filename = os.path.join(
        'data',
        f'dat_{args.current_seed}_{planner.update_counter}.combined.pgz')

    pickle_name = os.path.join(args.save_dir, data_filename)
    with gzip.GzipFile(pickle_name, 'wb', compresslevel=COMPRESS_LEVEL) as f:
        cPickle.dump(subgoal_training_datum, f, protocol=-1)

    with open(csv_full_path, 'a') as f:
        f.write(f'{data_filename}\n')

    return True, pickle_name


def get_csv_file_supervised(args):
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.supervised.csv'
    return os.path.join(args.save_dir, csv_filename)


def get_csv_file_combined(args):
    csv_filename = f'{args.data_file_base_name}_{args.current_seed}.combined.csv'
    return os.path.join(args.save_dir, csv_filename)
