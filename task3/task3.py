import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from Util import *


def linear_prediction(X_0, X_1, dt):
    V = (X_1 - X_0) / dt
    A = linear_approximation(X_0, V)
    NU = np.matmul(X_0, A)
    X_1_prediction = NU * dt + X_0
    mean_squared_error = mse(X_1, X_1_prediction)

    return X_1_prediction, mean_squared_error, V


def nonlinear_prediction(X_0, X_1, L, epsilon=0.1761680514483659):
    # D = distance_matrix(X_0)
    # np.sqrt(np.max(D)) * 0.05
    C, phis = nonlinear_approximation(X_0, X_1, epsilon, L, [])
    X_1_prediction = np.matmul(phis, C)
    mean_squared_error = mse(X_1, X_1_prediction)

    return X_1_prediction, mean_squared_error


def part1(X_0, X_1):
    """

    :param X_0: (2000, 2)
    :param X_1: (2000, 2)
    :return:
    """
    dt = 0.1
    X_1_prediction, mean_squared_error, V = linear_prediction(X_0, X_1, dt)

    # Plot true value and approximated values
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X_0[:, 0], X_0[:, 1], color='dodgerblue', s=2, label="X0")
    ax.scatter(X_1[:, 0], X_1[:, 1], color='indianred', s=2, label="X1")
    ax.scatter(X_1_prediction[:, 0], X_1_prediction[:, 1], color='black', s=2, label="X1 Prediction")
    ax.scatter(V[:, 0], V[:, 1], color='gold', s=2, label="Vector Field")
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_title('$\Delta t$ = {}, MSE = {}'.format(dt, mean_squared_error))
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.legend()
    plt.savefig('plots/task_3_part_1_approximation.png')
    plt.show()

    dts = np.linspace(0.00001, 1, 100000)
    mse = np.zeros(dts.shape)
    for i, dt in enumerate(dts):
        _, mse[i], _ = linear_prediction(X_0, X_1, dt)
        print("dt" + str(dt) + " mse[" + str(i) + "]:" + str(mse[i]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dts, mse, color='dodgerblue', s=2)
    plt.xlabel("$\Delta t$")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('plots/task_3_part_1_mse_vs_dt.png')
    plt.show()


def part2(X_0, X_1):
    epsilon = 0.1761680514483659

    # Plot true value and approximated values
    X_1_prediction_L_100, mse_L_100 = nonlinear_prediction(X_0, X_1, 100, epsilon)
    X_1_prediction_L_1000, mse_L_1000 = nonlinear_prediction(X_0, X_1, 1000, epsilon)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X_1[:, 0], X_1[:, 1], color='indianred', s=2, label="X1")
    ax.scatter(X_1_prediction_L_100[:, 0], X_1_prediction_L_100[:, 1], color='black', s=2,
               label="X_1_prediction(L=100, MSE={:.3f})".format(mse_L_100))
    ax.scatter(X_1_prediction_L_1000[:, 0], X_1_prediction_L_1000[:, 1], color='dodgerblue', s=2,
               label="X_1_prediction(L=1000, MSE={:.3f})".format(mse_L_1000))
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_title('$\epsilon$ = {}'.format(epsilon))
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.savefig('plots/task_3_part_2_approximation.png')
    plt.legend()
    plt.show()

    # Plot MSE vs L Graph
    Ls = np.arange(1, 2000, 1)
    mse = np.zeros(Ls.shape)
    for i, L in enumerate(Ls):
        _, mse[i] = nonlinear_prediction(X_0, X_1, L, epsilon)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(Ls, mse, color='indianred', s=2)
    plt.xlabel("L")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('plots/task_3_part_2_mse_vs_l.png')
    plt.show()

    """
    ?How do the errors differ to the linear approximation?
    >In the linear approximation error were not decreased.
    There was no linear function to approximate the given data with low error rate.
    However, when radial basis functions are used,  error are decreased with larger number of center points.

    # What do you conclude, is the vector field linear or nonlinear?

    # Why?
    """


def main():
    X_0 = read_file("nonlinear_vectorfield_data_x0.txt")  # (2000, 2)
    X_1 = read_file("nonlinear_vectorfield_data_x1.txt")  # (2000, 2)

    # part1(X_0, X_1)
    part2(X_0, X_1)


if __name__ == '__main__':
    main()
