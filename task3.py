import matplotlib.pyplot as plt
from Util import *


def read_file_0(file_path):
    """
    Read data from a given file then return the numpy array

    :param file_path: name of the file with extension.
    :return: [N, D] numpy array of the data in the file
    """

    file = open(file_path, "r")
    var = []
    for line in file:
        # TODO: float may cause casting issue. Check it!
        var.append(tuple(map(float, line.rstrip().split())))
    file.close()

    return np.array(var)



def vector_field_approx(X_0, X_1, dt=0.1):
    V = (X_1 - X_0) / dt
    A = linear_approximation(X_0, V)
    NU = np.matmul(X_0, A)
    X_1_prediction = NU * dt + X_0
    mean_squared_error = mse(X_1, X_1_prediction)
    return mean_squared_error





def main():
    X_0 = read_file_0("/Users/eysvr/Documents/praktikum/exercise_5/exercise5/data/nonlinear_vectorfield_data_x0.txt")  # (1000, 2)
    X_1 = read_file_0("/Users/eysvr/Documents/praktikum/exercise_5/exercise5/data/nonlinear_vectorfield_data_x1.txt")  # (1000, 2)
    #print(X_0)
    #print(X_1)


main()

