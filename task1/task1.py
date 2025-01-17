import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append('../')
from Util import *


def plot_linear_approximation(x, fx, fx_approximated):
    """
    Plots given linear approximation of a function.
    :param x: Data points
    :param fx: Function values of data points
    :param fx_approximated: Linear approximation of function values
    """
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, fx, color='indianred', s=2, label="Original Data")
    ax.plot(x, fx_approximated, c='dodgerblue', linewidth=0.5, label="Linear Approximation")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()


def plot_nonlinear_approximation(x, fx, fx_approximated, L, epsilon):
    """
    Plots given nonlinear approximation of a function.

    :param x: Data points
    :param fx: Function values of data points
    :param fx_approximated: Nonlinear approximation of function values
    :param L: Number of radial basis functions
    :param epsilon: Bandwidth
    """
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, fx, color='indianred', s=2, label="Original Data")
    ax.scatter(x, fx_approximated, color='dodgerblue', s=2, label="Nonlinear Approximation")
    ax.set_title('L = {}, $\epsilon$ = {:.3f}'.format(L, epsilon))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()


def plot_mse_vs_epsilon_and_l(D, fx_approximated_linear, fx_linear, x_linear_new):
    """
    Plots Mean Squared Error values for different L and bandwidth values.

    :param D: Distance matrix [N,N]
    :param fx_approximated_linear: Linear approximation values of function
    :param fx_linear: Real values of function
    :param x_linear_new: Data points of linear dataset
    """
    linear_approximation_error = mse(fx_linear, fx_approximated_linear)  # mse=1.0604702468531419e-10
    # Calculate MSE for each epsilon value
    L = 20
    es = np.linspace(0.05, 10, 100) * np.sqrt(np.max(D))
    mess = np.zeros(es.shape)
    for index, epsilon in enumerate(es):
        C, phis, xl = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L, [])
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
    epsilon = np.sqrt(np.max(D)) * 0.05
    for index, L in enumerate(Ls):
        C, phis, xl = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L, [])
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


def part_1():
    data_linear = read_file('linear_function_data.txt')  # (1000, 2)
    x_linear = data_linear[:, 0]
    fx_linear = data_linear[:, 1]
    x_linear_new = np.vstack([x_linear, np.ones(len(x_linear))]).T  # (1000, 2)
    linear_start_time = time.time()

    # y = ax + b
    # x_linear_new: (N, 2), first column has x's which will be multiplied by 'a' and,
    # the second column has 1's which will be multiplied by 'b'
    # AT: (2,), [a, b]
    # F = X.dot(A.T)
    AT = linear_approximation(x_linear_new, fx_linear)
    fx_approximated_linear = np.matmul(x_linear_new, AT)
    linear_cal_time = time.time() - linear_start_time
    plot_linear_approximation(x_linear, fx_linear, fx_approximated_linear)

    L = 100
    # Calculating epsilon as in Diffusion Map algorithm
    nonlinear_start_time = time.time()
    D = distance_matrix(x_linear)
    epsilon = np.sqrt(np.max(D)) * 0.05

    CT, phis, xl = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L, [])
    fx_approximated_nonlinear = np.matmul(phis, CT)
    nonlinear_cal_time = time.time() - nonlinear_start_time
    plot_nonlinear_approximation(x_linear, fx_linear, fx_approximated_nonlinear, L, epsilon)

    #"""
    #TEST for vector notation
    CT, phis, xl = nonlinear_approximation(x_linear, fx_linear, epsilon, L, [])
    fx_approximated_nonlinear = np.matmul(phis, CT)
    nonlinear_cal_time = time.time() - nonlinear_start_time
    plot_nonlinear_approximation(x_linear, fx_linear, fx_approximated_nonlinear, L, epsilon)
    #"""

    # Why is it not a good idea to use radial basis functions for dataset (A)?
    # 1. Computational cost
    print("Linear calculation time: {}, Nonlinear calculation Time: {}".format(linear_cal_time, nonlinear_cal_time))
    # Linear calculation time: 0.0005049705505371094, Nonlinear calculation Time: 18.813239097595215

    # 2. Hard to find parameters L and ε for best approximation
    plot_mse_vs_epsilon_and_l(D, fx_approximated_linear, fx_linear, x_linear_new)


def part_2():
    data_nonlinear = read_file('nonlinear_function_data.txt')
    print(data_nonlinear.shape)
    x_linear = data_nonlinear[:, 0]
    fx_linear = data_nonlinear[:, 1]
    x_linear_new = np.vstack([x_linear, np.ones(len(x_linear))]).T
    A = linear_approximation(x_linear_new, fx_linear)
    fx_approximated_linear = np.matmul(x_linear_new, A)
    plot_linear_approximation(x_linear, fx_linear, fx_approximated_linear)


def part_3():
    data_nonlinear = read_file('nonlinear_function_data.txt')
    x_nonlinear = data_nonlinear[:, 0]
    fx_nonlinear = data_nonlinear[:, 1]
    L = 10

    # Calculating epsilon as in Diffusion Map algorithm
    D = distance_matrix(x_nonlinear)
    epsilon = np.sqrt(np.max(D)) * 0.05

    x_nonlinear_new = np.vstack([x_nonlinear, np.ones(len(x_nonlinear))]).T
    C, phis, xl = nonlinear_approximation(x_nonlinear_new, fx_nonlinear[:, np.newaxis], epsilon, L, [])
    fx_approximated_nonlinear = np.matmul(phis, C)
    plot_nonlinear_approximation(x_nonlinear, fx_nonlinear, fx_approximated_nonlinear, L, epsilon)


def main():
    part_1()
    part_2()
    part_3()


if __name__ == '__main__':
    main()
