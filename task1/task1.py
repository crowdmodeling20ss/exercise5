import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


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


def part_1():
    data_linear = read_file('linear_function_data.txt')
    x_linear = data_linear[:, 0]
    fx_linear = data_linear[:, 1]
    x_linear_new = np.vstack([x_linear, np.ones(len(x_linear))]).T
    A = linear_approximation(x_linear_new, fx_linear)
    fx_approximated_linear = np.matmul(x_linear_new, A) # or A * x_linear ? did not change anything.
    plot_linear_approximation(x_linear, fx_linear, fx_approximated_linear)

    # Why is it not a good idea to use radial basis functions for dataset (A)?
    L = 10
    # Calculating epsilon as in Diffusion Map algorithm
    D = distance_matrix(x_linear)
    epsilon = np.sqrt(np.max(D)) * 0.05
    C, phis = nonlinear_approximation(x_linear_new, fx_linear[:, np.newaxis], epsilon, L)
    fx_approximated_nonlinear = np.matmul(phis, C)
    plot_nonlinear_approximation(x_linear, fx_linear, fx_approximated_nonlinear, L, epsilon)


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
