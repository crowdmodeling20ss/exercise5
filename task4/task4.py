import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from Util import read_file


def part_1():
    data = read_file('takens_1.txt')
    time = np.arange(0,data.shape[0],1)
    x0 =  data[:, 0]
    fig, ax = plt.subplots(1, 1)
    ax.plot(time, x0, c='dodgerblue', linewidth=0.5)
    ax.set_xlabel('time')
    ax.set_ylabel('$x_0$')
    plt.show()

    n_delay = 500
    delayed = np.hstack((x0[n_delay:], x0[:n_delay]))
    fig, ax = plt.subplots(1, 1)
    ax.plot(x0, delayed, c='dodgerblue', linewidth=0.5)
    plt.show()


def main():
    part_1()


if __name__ == '__main__':
    main()
