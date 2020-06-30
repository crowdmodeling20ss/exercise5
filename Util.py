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

    :param x:
    :return:
    """
    n = x.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(x[i] - x[j])

    return dist_matrix


def linear_approximation(x, fx):
    """

    :param x:
    :param fx:
    :return:
    """
    A, res, rnk, s = lstsq(x, fx)
    return A

def plot_mse_vs_epsilon_and_l_task5(fx_approximated_linear, fx_linear, x_linear_new):
    linear_approximation_error = mse(fx_linear, fx_approximated_linear)  # mse=1.0604702468531419e-10
    # Calculate MSE for each epsilon value
    L = 20
    es = np.linspace(0.1, 1, 1000)
    mess = np.zeros(es.shape)
    for index, epsilon in enumerate(es):
        C, phis = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L, [])
        fx_approximated_nonlinear = np.matmul(phis, C)

        nonlinear_approximation_error = mse(fx_linear, fx_approximated_nonlinear.reshape(-1))
        error_difference = nonlinear_approximation_error - linear_approximation_error
        print("epsilon: {}, linear error: {}, nonlinear error: {} difference: {}"
              .format(epsilon, linear_approximation_error, nonlinear_approximation_error, error_difference))
        mess[index] = nonlinear_approximation_error
    fig, ax = plt.subplots(1, 1)
    ax.scatter(es, mess, color='indianred', s=5, label='Nonlinear MSE')
    ax.plot(es, [linear_approximation_error] * len(es), linewidth=0.5, color='dodgerblue', label='Linear MSE')
    ax.set_title('L = {}'.format(L))
    ax.set_xlabel('$\epsilon$')
    ax.set_ylabel('MSE')
    plt.legend()
    plt.show()
    # Calculate MSE for each L value
    Ls = np.arange(1, 1000, 2)
    mess = np.zeros(Ls.shape)
    #epsilon = np.sqrt(np.max(D)) * 0.05
    epsilon = 0.8
    for index, L in enumerate(Ls):
        C, phis = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L)
        fx_approximated_nonlinear = np.matmul(phis, C)

        nonlinear_approximation_error = mse(fx_linear, fx_approximated_nonlinear.reshape(-1))
        error_difference = nonlinear_approximation_error - linear_approximation_error
        print("L: {}, linear error: {}, nonlinear error: {} difference: {}"
              .format(L, linear_approximation_error, nonlinear_approximation_error, error_difference))
        mess[index] = nonlinear_approximation_error
    fig, ax = plt.subplots(1, 1)
    ax.scatter(Ls, mess, color='indianred', s=5, label='Nonlinear MSE')
    ax.plot(Ls, [linear_approximation_error] * len(Ls), linewidth=0.5, color='dodgerblue', label='Linear MSE')
    ax.set_title('$\epsilon$ = {}'.format(epsilon))
    ax.set_xlabel('L')
    ax.set_ylabel('MSE')
    plt.legend()
    plt.show()



def nonlinear_approximation(x, fx, epsilon, L, xl):
    """

    :param x:
    :param fx:
    :param epsilon:
    :param L:
    :return:
    """
    number_of_rows = x.shape[0]
    if xl == []:
        random_indices = np.random.choice(number_of_rows, size=L, replace=False)
        xl = x[random_indices]

    # Radial basis functions
    phis = np.zeros((number_of_rows, L))
    for l in range(L):
        phis[:, l] = np.exp(-(np.linalg.norm(x - xl[l], axis=1) ** 2) / epsilon ** 2)

    C = linear_approximation(phis, fx)

    return C, phis, xl


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

