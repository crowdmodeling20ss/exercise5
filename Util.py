import numpy as np
import matplotlib.pyplot as plt
import time
import sys
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


def distance_matrix(x):
    """
    Creates distance matrix from given vector.

    :param x: Vector [N,].
    :return: Matrix [N,N]
    """
    n = x.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(x[i] - x[j])

    return dist_matrix


def linear_approximation(x, fx):
    """
    Linear function in matrix notation:
    F = XA.T

    'lstsq' finds and returns the p in equation:
    y = M.dot(p)

    Therefore 'lstsq' return transpose of matrix A for our case.

    :param x: [N, n] design matrix
    :param fx: [N, d] output matrix
    :return: [n, d] A.T
    """
    AT, res, rnk, s = lstsq(x, fx)
    return AT



def nonlinear_approximation(x, fx, epsilon, L, xl):
    """""

    :param x: (N, n) input values
    :param fx: (N, d) output values
    :param epsilon: bandwidth
    :param L: number of basis functions
    :param xl: (L, n) predefined random points for basis functions.
    :return:
    """""
    number_of_rows = x.shape[0]
    if xl == []:
        random_indices = np.random.choice(number_of_rows, size=L, replace=False)
        xl = x[random_indices]  # xl ∈ R^n

    # Radial basis functions
    phis = np.zeros((number_of_rows, L))
    for l in range(L):
        phis[:, l] = np.exp(-(np.linalg.norm(matrix(x) - xl[l], axis=1) ** 2) / epsilon ** 2)

    CT = linear_approximation(phis, fx)  # C.T = (L, d)

    return CT, phis, xl


def lorenzEquations(t, x0, sigma, rho, beta):
    """
    Computes result of Lorenz Equation given its parameters.

    :param t: One-dimensional independent variable (time)
    :param x0: Starting points of the system.
    :param sigma: Parameter sigma of the function.
    :param rho: Parameter rho of the function.
    :param beta: Parameter beta of the function.
    :return: Calculated value of the function.
    """
    x, y, z = x0
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return dxdt, dydt, dzdt

