"""
Example code for creating a figure using matplotlib to display tensorflow model performance

Jay Rothenberger (jay.c.rothenberger@ou.edu)
"""


import fnmatch
import os
import re

import matplotlib.pyplot as plt
import pickle
import numpy as np


def read_all_pkl(dirname, filebase):
    """
    Read results from dirname from files matching filebase

    :param dirname: directory to read .pkl files from
    :param filebase: prefix that all files to read start with
    :@ return: a list of pickle loaded objects
    """

    # The set of files in the directory
    files = [f for f in os.listdir(dirname) if re.match(r'%s.+.pkl' % filebase, f)]
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s" % (dirname, f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)

    return results


def figure_metric_epochs(metric, name, fname, filebase=''):
    """
    builds and saves a figure of accuracy v.s. epochs as fname

    :param metric: name of the metric from history to plot v.s. epochs.  'val_binary_accuracy' etc.
    :param name: label that will appear on the plot before 'Accuracy' either 'Validation' or 'Testing' or 'Training'
    :param fname: name of file that we will save the figure to
    """
    results = read_all_pkl('results/', filebase)

    legend = [f'experiment {result["args"].filters}' for result in results]

    for result in results:
        # plot the metric v.s. epochs for each model
        series = result['history'][metric]
        plt.plot(range(len(series)), series, linestyle='-')

    # add the plot readability information
    plt.title(f'{name} as a Function of Epochs')
    plt.legend(legend)
    plt.xlabel('epochs')

    # save the figure
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.savefig(fname)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    figure_metric_epochs('loss', 'Train Loss', 'Figure 1', filebase='MNIST')
    figure_metric_epochs('sparse_categorical_accuracy', 'Train Accuracy', 'Figure 2', filebase='MNIST')
    figure_metric_epochs('val_sparse_categorical_accuracy', 'Validation Accuracy', 'Figure 3', filebase='MNIST')
