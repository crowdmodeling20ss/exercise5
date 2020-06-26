import numpy as np
from scipy.linalg import lstsq

def read_file(file_path):
    """
    Read data from a given file then return the numpy array

    :param file_path: name of the file with extension.
    :return: [N, D] numpy array of the data in the file
    """

    file = open("../data/" + file_path, "r")
    var = []
    for line in file:
        # TODO: float may cause casting issue. Check it!
        var.append(tuple(map(float, line.rstrip().split())))
    file.close()

    return np.array(var)


def mse(y_truth, y_pred):
    """
    Also sklearn.metrics.mean_squared_error can be used.

    :param y_truth: True data from observation
    :param y_pred: Approximation of the data
    :return: minimum squared root error between true data and approximation
    """
    return ((y_truth - y_pred) ** 2).mean()


def linear_approximation(x, fx):
    """

    :param x:
    :param fx:
    :return:
    """
    A, res, rnk, s = lstsq(x, fx)
    return A
