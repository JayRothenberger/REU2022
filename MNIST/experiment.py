"""
DLCC example experiment code
Author: Jay C. Rothenberger (jay.c.rothenberger@ou.edu)

"""
import os
import sys
import argparse
import pickle
import numpy as np

from job_control import *
from cnn_network import *


def train_validation_split(x_train, y_train, val_fraction=.2, shuffle=True):
    # generate a random permutation of the indicies (shuffle the data)
    # this ensures the distribution of examples is the same along the split
    indices = np.random.permutation(x_train.shape[0])
    # calculate the index where we wish to split
    split_index = int(x_train.shape[0] * val_fraction)
    # splite the permutation into sets of the desired size
    train_indices, val_indices = indices[split_index:], indices[:split_index]
    # return (x_train, y_train), (x_val, y_val)
    return (x_train[train_indices], y_train[train_indices]), (x_train[val_indices], y_train[val_indices])


def prepare_data_set():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
    (x_train, y_train), (x_val, y_val) = train_validation_split(x_train, y_train)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_parser():
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')

    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--label', type=str, default="", help="Experiment label")
    parser.add_argument('--exp_type', type=str, default='MNIST', help="Experiment type")
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp', type=int, default=0, help='Experiment index')
    parser.add_argument('--fold', type=int, default=1, help='training fold')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=1e-3, help="Learning rate")

    # convolutional unit parameters
    parser.add_argument('--filters', nargs='+', type=int, default=[],
                        help='Number of filters per layer (sequence of ints)')
    parser.add_argument('--kernels', nargs='+', type=int, default=[],
                        help='kernel sizes for each layer layer (sequence of ints)')

    # Hidden unit parameters
    parser.add_argument('--hidden', nargs='+', type=int, default=[],
                        help='Number of hidden units per layer (sequence of ints)')
    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=25, help="Patience for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=1024, help="Training set batch size")

    return parser


def exp_type_to_hyperparameters(args):
    """
    convert the type of experiment to the appropriate hyperparameter grids

    :param args: the CLA for the experiment.  Expects an element 'exp_type' \in {'BLSTM', 'GRU'}

    @return a dictionary of hyperparameters
    """

    params = {
        'filters': [],
        'kernels': [],
        'hidden': []
    }

    return params


def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp
    if (index is None):
        return ""

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())

    # Check bounds
    assert (args.exp >= 0 and args.exp < ji.get_njobs()), "exp out of range"

    # Print the parameters specific to this exp
    print(ji.get_index(args.exp))

    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp, args)


def generate_fname(args, params_str):
    """
    Generate the base file name for output files/directories.

    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    """
    # network parameters
    hidden_str = '_'.join([str(i) for i in args.hidden])
    filters_str = '_'.join([str(i) for i in args.filters])
    kernels_str = '_'.join([str(i) for i in args.kernels])

    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_" % args.label

    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_" % args.exp_type

    # experiment index
    num_str = str(args.exp)

    # learning rate
    lrate_str = "LR_%0.6f_" % args.lrate

    # Put it all together, including # of training folds and the experiment rotation
    return "%s/%s_%s_filt_%s_ker_%s_hidden_%s" % (
        args.results_path,
        experiment_type_str,
        num_str,
        filters_str,
        kernels_str,
        hidden_str)


def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model

    :param args: Argparse arguments
    '''

    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    print(args.exp)

    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    # Split metadata into individual data sets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data_set()
    # arguments that are passed to each of the network returning functions
    network_params = {'learning_rate': args.lrate,
                      'conv_filters': args.filters,
                      'conv_size': args.kernels,
                      'dense_layers': args.hidden,
                      'image_size': x_train[0].shape,
                      'n_classes': 10}

    # Build network: you must provide your own implementation
    model = None  # TODO

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.pkl" % fbase

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists" % fname_out)
        return

    # Callbacks
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=args.patience,
                                                         restore_best_weights=True,
                                                         min_delta=args.min_delta)

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #  validation_steps=None means that ALL validation samples will be used (of the selected subset)
    history = model.fit(x_train, y_train,
                        batch_size=args.batch,
                        epochs=args.epochs,
                        verbose=True,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping_cb])

    # Generate results data
    results = dict()

    results['args'] = args

    # evaluate the model on the different sets
    results['validation_eval'] = model.evaluate(x_val, y_val, return_dict=True)
    results['training_eval'] = model.evaluate(x_train, y_train, return_dict=True)
    results['test_eval'] = model.evaluate(x_test, y_test, return_dict=True)

    results['history'] = history.history

    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl" % (fbase), "wb") as fp:
        pickle.dump(results, fp)

    # (don't) save model
    # model.save("%s_model"%(fbase))

    print(fbase)

    return model


if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # just use one GPU
        pass  # (do nothing)

    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU')
    n_physical_devices = len(physical_devices)

    if n_physical_devices > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')

    execute_exp(args)
