import matplotlib.pyplot as plt

from Util import *


def part1(X_0, X_1):
    # estimate the linear vector field that was used to generate the points x1 from the points x
    # Use the finite-difference formula from section (1.3) to estimate the vectors v(k) at all points x(k)
    # then approximate the matrix A ∈ R2×2 with a supervised learning problem
    V = (X_1 - X_0) / 0.1

    A = linear_approximation(X_0, V)
    NU = np.matmul(X_0, A)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(X_0[:, 0], X_0[:, 1], color='indianred', s=2, label="X0")
    ax.scatter(X_1[:, 0], X_1[:, 1], color='dodgerblue', s=2, label="X1")
    ax.scatter(V[:, 0], V[:, 1], color='black', s=2, label="V")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.savefig('plots/task_2_part_1.png')
    plt.show()

    return A


def part2(X_0, X_1, A):
    NU = np.matmul(X_0, A)
    X_1_prediction = NU * 0.1 + X_0

    mean_squared_error = mse(X_1, X_1_prediction)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(X_0[:, 0], X_0[:, 1], color='indianred', s=2, label="X0")
    ax.scatter(X_1[:, 0], X_1[:, 1], color='dodgerblue', s=2, label="X1")
    ax.scatter(X_1_prediction[:, 0], X_1_prediction[:, 1], color='orange', s=2, label="X1 Prediction")
    ax.set_title('MSE = {}'.format(mean_squared_error))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.savefig('plots/task_2_part_2.png')
    plt.show()

    return X_1_prediction, mean_squared_error


def part3(A):
    dt = 0.1
    T = 100
    x0 = np.array([10, 10])

    # Find trajectory
    trajectory = np.zeros((int(T / dt), len(x0)))
    for i, t in enumerate(np.arange(0, T, dt)):
        Vk = np.matmul(A, x0)
        x1 = (Vk * dt) + x0
        trajectory[i] = x1
        x0 = x1

    # Calculate phase portrait
    x = np.arange(-10, 10, 0.01)
    x1, x2 = np.meshgrid(x, x)
    y1 = A[0][0] * x1 + A[1][0] * x2
    y2 = A[0][1] * x1 + A[1][1] * x2

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(trajectory[:, 0], trajectory[:, 1], color='purple', s=2, label="trajectory")
    ax.streamplot(x1, x2, y1, y2, color='dodgerblue', linewidth=1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig('plots/task_2_part_3.png')
    plt.show()

    return trajectory, [x1, x2, y1, y2]


def main():
    X_0 = read_file("linear_vectorfield_data_x0.txt")  # (1000, 2)
    X_1 = read_file("linear_vectorfield_data_x1.txt")  # (1000, 2)
    A = part1(X_0, X_1)
    X_1_prediction, mean_squared_error = part2(X_0, X_1, A)
    trajectory, phase_portrait = part3(A)

    # show all in one plot
    fig, ax = plt.subplots(1, 1)
    ax.scatter(X_0[:, 0], X_0[:, 1], color='indianred', s=2, label="X0")
    ax.scatter(X_1[:, 0], X_1[:, 1], color='dodgerblue', s=2, label="X1")
    # ax.scatter(V[:, 0], V[:, 1], color='green', s=2, label="V")
    ax.scatter(X_1_prediction[:, 0], X_1_prediction[:, 1], color='orange', s=2, label="X1 Prediction")
    ax.scatter(trajectory[:, 0], trajectory[:, 1], color='purple', s=2, label="trajectory")
    ax.streamplot(*phase_portrait, color='gray', linewidth=1)
    ax.set_title('MSE = {}'.format(mean_squared_error))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.savefig('plots/task_2_all.png')
    plt.show()


if __name__ == '__main__':
    main()
