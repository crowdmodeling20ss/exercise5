import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import time


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
    return ((y_truth - y_pred) ** 2).mean(axis=0)


def distance_matrix(x):
    n = x.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(x[i] - x[j])

    return dist_matrix


def linear_approximation(x, fx):
    A, res, rnk, s = lstsq(x, fx)
    return A


def plot_linear_approximation(x, fx, fx_approximated):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, fx, color='indianred', s=2, label="Original Data")
    ax.plot(x, fx_approximated, c='dodgerblue', linewidth=0.5, label="Linear Approximation")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()


def plot_nonlinear_approximation(x, fx, fx_approximated, L, epsilon):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, fx, color='indianred', s=2, label="Original Data")
    ax.scatter(x, fx_approximated, color='dodgerblue', s=2, label="Nonlinear Approximation")
    ax.set_title('L = {}, $\epsilon$ = {:.3f}'.format(L, epsilon))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()


def nonlinear_approximation(x, fx, epsilon, L):
    number_of_rows = x.shape[0]
    random_indices = np.random.choice(number_of_rows, size=L, replace=False)
    xl = x[random_indices]

    # Radial basis functions
    phis = np.zeros((number_of_rows, L))
    for l in range(L):
        phis[:, l] = np.exp(-(np.linalg.norm(x - xl[l], axis=1) ** 2) / epsilon ** 2)

    C = linear_approximation(phis, fx)

    return C, phis


def plot_mse_vs_epsilon_and_l(D, fx_approximated_linear, fx_linear, x_linear_new):
    linear_approximation_error = mse(fx_linear, fx_approximated_linear)  # mse=1.0604702468531419e-10
    # Calculate MSE for each epsilon value
    L = 20
    es = np.linspace(0.05, 10, 100) * np.sqrt(np.max(D))
    mess = np.zeros(es.shape)
    for index, epsilon in enumerate(es):
        C, phis = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L)
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
        C, phis = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L)
        fx_approximated_nonlinear = np.matmul(phis, C)

        nonlinear_approximation_error = mse(fx_linear, fx_approximated_nonlinear.reshape(-1))
        error_difference = nonlinear_approximation_error - linear_approximation_error
        print("epsilon: {}, linear error: {}, nonlinear error: {} difference: {}"
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
    A = linear_approximation(x_linear_new, fx_linear)  # (2, )
    fx_approximated_linear = np.matmul(x_linear_new, A)  # or A * x_linear ? did not change anything.
    linear_cal_time = time.time() - linear_start_time
    plot_linear_approximation(x_linear, fx_linear, fx_approximated_linear)

    L = 10
    # Calculating epsilon as in Diffusion Map algorithm
    nonlinear_start_time = time.time()
    D = distance_matrix(x_linear)
    epsilon = np.sqrt(np.max(D)) * 0.05
    C, phis = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L)
    fx_approximated_nonlinear = np.matmul(phis, C)
    nonlinear_cal_time = time.time() - nonlinear_start_time
    plot_nonlinear_approximation(x_linear, fx_linear, fx_approximated_nonlinear, L, epsilon)

    # Why is it not a good idea to use radial basis functions for dataset (A)?
    # 1. Computational cost
    print("Linear calculation time: {}, Nonlinear calculation Time: {}".format(linear_cal_time, nonlinear_cal_time))
    # Linear calculation time: 0.0005049705505371094, Nonlinear calculation Time: 18.813239097595215

    # 2. Hard to find parameters L and ε for best approximation
    plot_mse_vs_epsilon_and_l(D, fx_approximated_linear, fx_linear, x_linear_new)


def part_2():
    data_nonlinear = read_file('nonlinear_function_data.txt')
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
    C, phis = nonlinear_approximation(x_nonlinear_new, fx_nonlinear[:, np.newaxis], epsilon, L)
    fx_approximated_nonlinear = np.matmul(phis, C)
    plot_nonlinear_approximation(x_nonlinear, fx_nonlinear, fx_approximated_nonlinear, L, epsilon)


def main():
    part_1()
    part_2()
    part_3()


if __name__ == '__main__':
    main()
