import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from Util import *
import time


def linear_prediction(X_0, X_1, dt):
    V = (X_1 - X_0) / dt
    AT = linear_approximation(X_0, V)
    NU = np.matmul(X_0, AT)
    X_1_prediction = NU * dt + X_0
    mean_squared_error = mse(X_1, X_1_prediction)

    return X_1_prediction, mean_squared_error, NU, AT


def nonlinear_prediction(X_0, X_1, L, dt, epsilon=0.1761680514483659):
    # D = distance_matrix(X_0)
    # np.sqrt(np.max(D)) * 0.05
    V = (X_1 - X_0) / dt  # ???
    CT, phis, xl = nonlinear_approximation(X_0, V, epsilon, L, [])
    NU = np.matmul(phis, CT)
    X_1_prediction = NU * dt + X_0
    mean_squared_error = mse(X_1, X_1_prediction)

    return X_1_prediction, NU, mean_squared_error


def part1(X_0, X_1):
    """

    :param X_0: (2000, 2)
    :param X_1: (2000, 2)
    :return:
    """
    dt = 0.1
    X_1_prediction, mean_squared_error, V, AT = linear_prediction(X_0, X_1, dt)
    # A = [[-0.100, -0.002],[0.009, -0.433]]

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

    dts = np.linspace(0.000000000001, 1.1, 100)
    mses = np.zeros(dts.shape)
    for i, dt in enumerate(dts):
        # _, mse[i], _ = linear_prediction(X_0, X_1, dt)
        X_1_prediction = np.matmul(X_0, AT) * dt + X_0
        mses[i] = mse(X_1, X_1_prediction)
        print("dt" + str(dt) + " mse[" + str(i) + "]:" + str(mses[i]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dts, mses, color='dodgerblue', s=2)
    plt.xlabel("$\Delta t$")
    plt.ylabel("MSE")
    plt.savefig('plots/task_3_part_1_mse_vs_dt.png')
    plt.show()
    # Minimum MSE = 0.037270081174962, for mses[9]. dts[9] = 0.1000000

    # dt does not affect the mean square error


def part2(X_0, X_1):
    epsilon = 0.1761680514483659
    dt = 0.1

    # Plot true value and approximated values
    X_1_prediction_L_100, V_L_100, mse_L_100 = nonlinear_prediction(X_0, X_1, 100, dt, epsilon)
    X_1_prediction_L_1000, V_L_1000, mse_L_1000 = nonlinear_prediction(X_0, X_1, 1000, dt, epsilon)

    plot_task3_non_linear_approximation(V_L_1000, X_1, X_1_prediction_L_100, X_1_prediction_L_1000, dt, epsilon,
                                        mse_L_100, mse_L_1000)

    L = 100
    V = (X_1 - X_0) / dt
    # Once we calculate CT they will not be changed but we will use it to approximate X_1
    # nonlinear_approximation already return 'phis' for X_0
    CT, phis, xl = nonlinear_approximation(X_0, V, epsilon, L, [])

    # If we want to use different initial value to approximate X_1, we can do it like below
    X_0_new = X_0
    phis, _ = calculate_phis(L, epsilon, X_0_new, xl)
    NU = np.matmul(phis, CT)  # approximated vector field
    X_1_prediction = NU * dt + X_0

    # Let's do the same example that is done with linear approximation function
    dts = np.linspace(0.000000000001, 1.1, 100)
    mses = np.zeros(dts.shape)
    msex = np.zeros(dts.shape)
    for i, dt in enumerate(dts):
        X_1_prediction, _, _ = nonlinear_prediction(X_0, X_1, L, dt, epsilon)
        msex[i] = mse(X_1, X_1_prediction)
        X_1_prediction = NU * dt + X_0
        mses[i] = mse(X_1, X_1_prediction)
        print("dt" + str(dt) + " mse[" + str(i) + "]:" + str(mses[i]))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dts, mses, color='dodgerblue', s=2)
    # ax.scatter(dts, msex, color='red', s=2)
    plt.xlabel("$\Delta t$")
    plt.ylabel("MSE")
    plt.savefig('plots/task_3_part_2_mse_vs_dt.png')
    plt.show()

    '''
    # Plot MSE vs L Graph
    Ls = np.arange(1, 2000, 1)
    mses = np.zeros(Ls.shape)
    for i, L in enumerate(Ls):
        _, _, mses[i] = nonlinear_prediction(X_0, X_1, L, epsilon)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(Ls, mses, color='indianred', s=2)
    plt.xlabel("L")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('plots/task_3_part_2_mse_vs_l.png')
    plt.show()
    '''

    """
    ?How do the errors differ to the linear approximation?
    >In the linear approximation error were not decreased.
    There was no linear function to approximate the given data with low error rate.
    However, when radial basis functions are used,  error are decreased with larger number of center points.

    # What do you conclude, is the vector field linear or nonlinear?

    # Why?
    """
    plot_part3_mse_vs_dt(V_L_1000, mse_L_1000, X_0, X_1, start=0.000000000001, stop=0.3)
    plot_part3_mse_vs_dt(V_L_1000, mse_L_1000, X_0, X_1, start=0.000000000001, stop=1)


def plot_task3_non_linear_approximation(V_L_1000, X_1, X_1_prediction_L_100, X_1_prediction_L_1000, dt, epsilon,
                                        mse_L_100, mse_L_1000):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X_1[:, 0], X_1[:, 1], color='indianred', s=2, label="X1")
    ax.scatter(V_L_1000[:, 0], V_L_1000[:, 1], color='gold', s=2, label="Vector field")
    ax.scatter(X_1_prediction_L_100[:, 0], X_1_prediction_L_100[:, 1], color='black', s=2,
               label="X_1_prediction(L=100, MSE={:.3f})".format(mse_L_100))
    ax.scatter(X_1_prediction_L_1000[:, 0], X_1_prediction_L_1000[:, 1], color='dodgerblue', s=2,
               label="X_1_prediction(L=1000, MSE={:.3f})".format(mse_L_1000))
    ax.set_xlim([-4.5, 4.5])
    ax.set_ylim([-4.5, 4.5])
    ax.set_title('$\Delta t$ = {}, $\epsilon$ = {}'.format(dt, epsilon))
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.legend()
    plt.savefig('plots/task_3_part_2_approximation.png')
    plt.show()


def plot_part3_mse_vs_dt(V_L_1000, mse_L_1000, X_0, X_1, start, stop):
    dts = np.linspace(start, stop, 100)
    mses = np.zeros(dts.shape)
    for i, dt in enumerate(dts):
        X_1_prediction_L_1000 = V_L_1000 * dt + X_0
        mses[i] = mse(X_1, X_1_prediction_L_1000)
        print("dt" + str(dt) + " mse[" + str(i) + "]:" + str(mses[i]))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(dts, mses, color='dodgerblue', s=2)
    ax.set_title('Vector Field for L=1000, MSE = {:.3f}'.format(mse_L_1000))
    plt.xlabel("$\Delta t$")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('plots/task_3_part_3_mse_vs_dt' + str(start) + '_' + str(stop) + '.png')
    plt.show()


def part3(X_0, X_1):
    epsilon = 0.1761680514483659
    dt = 0.1
    L = 1000

    V = (X_1 - X_0) / dt
    CT, phis, xl = nonlinear_approximation(X_0, V, epsilon, L, [])

    task3_part_3_plot_steady_states(CT, L, X_0, X_1, dt, epsilon, xl)

    # Calculate phase portrait
    task3_part3_plot_phase_portrait(CT, L, epsilon, xl)

    # Plot trajectory starting from 0,0
    plot_trajectory(CT, L, epsilon, xl)


def plot_trajectory(CT, L, epsilon, xl):
    fig = plt.figure()
    ax = fig.add_subplot()
    T = 100
    dt = 0.1
    x0 = np.array([[0, 0]])
    # Find trajectory
    trajectory = np.zeros((int(T / dt), 2))
    for i, t in enumerate(np.arange(0, T, dt)):
        phis, _ = calculate_phis(L, epsilon, x0, xl)
        x0 = (np.matmul(phis, CT) * dt) + x0
        trajectory[i] = x0
    ax.scatter(trajectory[:, 0], trajectory[:, 1], cmap='GnBu', s=2, label="Trajectory")
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 2")
    plt.colorbar()
    ax.set_title('$\Delta t$ = {}, L = {}'.format(dt, L))
    plt.savefig('plots/task_3_part_3_trajectory_start_from_0_0.png')
    plt.show()


def task3_part3_plot_phase_portrait(CT, L, epsilon, xl):
    dt = 0.1
    # psi(x) = PHI(X) C.T
    x = np.arange(-10, 10, 0.01)
    x1, x2 = np.meshgrid(x, x)  # x1,x2 : 2000x2000
    # (4000000, 2)
    # phis=   (4000000, 100)
    # Y = (4000000, 2)
    Y = calcualate_next_points(CT, L, dt, epsilon, x1, x2, xl)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.streamplot(x1, x2, Y[:, 0].reshape(x1.shape), Y[:, 1].reshape(x1.shape), color='dodgerblue', linewidth=1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.savefig('plots/task_3_part_3.png')
    plt.show()


def task3_part_3_plot_steady_states(CT, L, X_0, X_1, dt, epsilon, xl):
    fig = plt.figure()
    ax = fig.add_subplot()
    T = 100
    x0 = X_0
    for i, t in enumerate(np.arange(0, T, dt)):
        phis, _ = calculate_phis(L, epsilon, x0, xl)
        x0 = (np.matmul(phis, CT) * dt) + x0
        error = mse(X_1, x0)

        plt.cla()
        ax.scatter(x0[:, 0], x0[:, 1], color='dodgerblue', s=2, label="x")
        plt.xlabel("coordinate 1")
        plt.ylabel("coordinate 2")
        ax.set_title('Vector Field for L={}, MSE = {:.3f}, step={}'.format(L, error, i))
        plt.pause(0.01)
    plt.show()


def calcualate_next_points(CT, L, dt, epsilon, x1, x2, xl):
    phis, _ = calculate_phis(L, epsilon, np.vstack((x1.flatten(), x2.flatten())).T, xl)
    NU = np.matmul(phis, CT)  # approximated vector field
    Y = NU * dt + np.vstack((x1.flatten(), x2.flatten())).T
    return Y


def main():
    X_0 = read_file("nonlinear_vectorfield_data_x0.txt")  # (2000, 2)
    X_1 = read_file("nonlinear_vectorfield_data_x1.txt")  # (2000, 2)

    part1(X_0, X_1)
    part2(X_0, X_1)
    part3(X_0, X_1)


if __name__ == '__main__':
    main()
