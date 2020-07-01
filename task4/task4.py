import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
from Util import read_file, lorenzEquations
from scipy.integrate import solve_ivp


def plotLorenz(real_coor, delayed_coor_1, delayed_coor_2, delay, axis=0):
    """
    Plots Lorenz Attractor with its given (delayed) coordinates.

    :param real_coor: Real x, y or z coordinate of the attractor.
    :param delayed_coor_1:
    :param delayed_coor_2:
    :param delay:
    :param axis:
    """
    fig = plt.figure()
    ax0 = fig.gca(projection='3d')
    ax0.plot(real_coor, delayed_coor_1, delayed_coor_2, linewidth=0.5, color="indianred", linestyle=':',
             antialiased=True)
    if axis == 0:
        ax0.set_xlabel('$x(t)$')
        ax0.set_ylabel('$x(t+ \Delta t)$')
        ax0.set_zlabel('$x(t+ 2 \Delta t)$')
    elif axis == 1:
        ax0.set_xlabel('$y(t)$')
        ax0.set_ylabel('$y(t+ \Delta t)$')
        ax0.set_zlabel('$y(t+ 2 \Delta t)$')
    else:
        ax0.set_xlabel('$z(t)$')
        ax0.set_ylabel('$z(t+ \Delta t)$')
        ax0.set_zlabel('$z(t+ 2 \Delta t)$')

    ax0.set_title("Lorenz Attractor, $\Delta t = {}$".format(delay))
    plt.show()


def part_2():
    time = (0.0, 1000.0)
    delta_t = 0.8
    t_eval = np.arange(0,1000,delta_t)
    x0 = [10, 10, 10]
    sigma = 10
    beta = 8.0 / 3
    rho = 28

    solution = solve_ivp(lorenzEquations, time, x0, t_eval=t_eval, args=(sigma, rho, beta,))
    sol = solution.y

    # plot for x
    x_real_coor = sol[0, :sol.shape[1]-2]
    x_delayed_1 = sol[0, 1:sol.shape[1]-1]
    x_delayed_2 = sol[0, 2:]
    plotLorenz(x_real_coor, x_delayed_1, x_delayed_2, delta_t, axis=0)

    #plot for y
    y_real_coor = sol[1, :sol.shape[1] - 2]
    y_delayed_1 = sol[1, 1:sol.shape[1] - 1]
    y_delayed_2 = sol[1, 2:]
    plotLorenz(y_delayed_2, y_real_coor, y_delayed_1, delta_t, axis=1)

    # plot for z
    z_real_coor = sol[2, :sol.shape[1]-2]
    z_delayed_1 = sol[2, 1:sol.shape[1]-1]
    z_delayed_2 = sol[2, 2:]
    plotLorenz(z_delayed_2, z_delayed_1, z_real_coor, delta_t, axis=2)


def part_1():
    data = read_file('takens_1.txt')
    time = np.arange(0,data.shape[0],1)
    x0 = data[:, 0]

    # Plotting the data
    fig, ax = plt.subplots(1, 1)
    ax.plot(x0, data[:, 1], c='dodgerblue', linewidth=0.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()

    # Plotting the first coordinate against the line number in the dataset (the “time”)
    fig, ax = plt.subplots(1, 1)
    ax.plot(time, x0, c='dodgerblue', linewidth=0.5)
    ax.set_xlabel('time')
    ax.set_ylabel('$x$')
    plt.show()

    # Plotting the coordinate against its delayed version , 2 dimensional
    n_delay = 50
    #delayed = np.hstack((x0[-n_delay:], x0[:data.shape[0]-n_delay]))
    x = x0[:x0.shape[0] - n_delay]
    delayed = x0[n_delay:]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, delayed, c='dodgerblue', linewidth=0.5) # x0
    ax.set_xlabel('$x(t)$')
    ax.set_ylabel('$x(t+ \Delta n)$')
    ax.set_title('$\Delta n = {}$'.format(n_delay))
    plt.show()

    # Plotting the coordinate against its delayed version , 3 dimensional
    n_delay_2 = 2 * n_delay
    x_2 = x0[:x0.shape[0] - n_delay_2]
    delayed_1 = x0[n_delay:x0.shape[0] - n_delay]
    delayed_2 = x0[n_delay_2:]
    fig = plt.figure()
    ax0 = fig.gca(projection='3d')
    ax0.plot(x_2, delayed_1, delayed_2, c='dodgerblue', linestyle=':', antialiased=True)
    ax0.set_xlabel('$x(t)$')
    ax0.set_ylabel('$x(t+ \Delta n)$')
    ax0.set_zlabel('$x(t+ 2 \Delta n)$')
    ax0.set_title('$\Delta n = {}$'.format(n_delay))
    plt.show()

def main():
    part_1()
    part_2()


if __name__ == '__main__':
    main()
