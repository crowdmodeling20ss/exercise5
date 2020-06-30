import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
from Util import *

from scipy.integrate import solve_ivp
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import math
#sys.path.append('../task1/')
#from task1 import plot_mse_vs_epsilon_and_l

def embed_data(df, np_shape = 13651):
    
    df = df[1000:]
    df = df.iloc[: , 1:4]
    data_np = df.to_numpy()
    embedded_np = np.empty((np_shape,1053))
    #print((data_np[0:351].flatten()).shape)
    for i in range(np_shape):
        embedded_np[i] = (data_np[i:(i+351)]).flatten()
    #print(embedded_np.shape)
    return embedded_np


def plot_column(df, col):
    data_np = df.to_numpy()
    plt.figure()
    for c in col:
        plt.plot(data_np[:,c])
    plt.show()


def plot_pca(pca_np):
    plt.figure()
    for i in range(pca_np.shape[1]):
        plt.plot(pca_np[:,i])
    plt.show()


def plotEmbedding(df, x0, x1, x2, c_array, axis=0):
    df = df[1000:]
    #df = df.iloc[: , c_array]
    data_np = df.to_numpy()
    data_np = data_np[:13651, c_array]
    #print(data_np.shape)
    fig = plt.figure()
    ax0 = fig.gca(projection='3d')
    ax0.scatter(x0, x1, x2, linewidth=1.5, c= data_np, linestyle=':',
             antialiased=True, cmap = "Spectral")
    #ax0.set_title("Lorenz Attractor, $\Delta t = {}$".format(delay))
    plt.show()
    #plt.close(5)


def plot_embedding_period(x0, x1, x2, axis=0):
    fig = plt.figure()

    c_list = range(0, (x0.shape[0]))

    ax0 = fig.gca(projection='3d')
    ax0.scatter(x0, x1, x2, linewidth=0.1, c=c_list, linestyle=':',
             antialiased=True, cmap = "viridis")
    #ax0.set_title("Lorenz Attractor, $\Delta t = {}$".format(delay))
    plt.show()
    #plt.close(5)



def color_coding(df, pca_np):
    for i in range(9):
        plotEmbedding(df, pca_np[:,0], pca_np[:,1], pca_np[:,2], i)


def print_gradient_list(grad_list):
    plt.plot(grad_list)
    plt.show()


def grad_approx_vis(gradient_list, grad_approx, L, epsilon):
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(gradient_list.shape[0]), gradient_list, color='indianred', linewidth=5.0, label="Original Data")
    ax.plot(range(gradient_list.shape[0]), grad_approx, color='dodgerblue', label="Nonlinear Approximation")
    ax.set_title('L = {}, $\epsilon$ = {:.3f}'.format(L, epsilon))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()


def plot_arclen_all_periods(total_grad_list):
    fig, ax = plt.subplots(1, 1)
    ax.plot(total_grad_list, color='indianred', label="Original Data")
    ax.set_title("Arclen of curves for 14 days")
    ax.set_xlabel('timesteps')
    ax.set_ylabel('arclen')
    plt.legend()
    plt.show()



def main():
    df = pd.read_csv('../data/MI_timesteps.txt', sep=" ")
    e_np = embed_data(df)
    print("embed shape:", e_np.shape)
    pca = PCA(n_components=3)
    pca.fit(e_np)
    pca_np = pca.transform(e_np)
    
    #The nine color plots:
    #color_coding(df, pca_np)
    
    
    #After many plots, we identified the period_size as "1997".
    
    reduc_pca = pca_np[:1997]#1997

    #we create two lists, one for arclength of curve at each point
    arclen_list = np.zeros((reduc_pca.shape[0]))
    vector_field = np.zeros((reduc_pca.shape[0], reduc_pca.shape[1]))
    for i in range(reduc_pca.shape[0]):
        if i == 0: #arc length at starting point is 0
            arclen_list[i] = 0
            vector_field[i] = 0
        else:
            vector_field[i] = (reduc_pca[i] - reduc_pca[i-1])
            arclen_list[i] = np.linalg.norm(vector_field[i]) + arclen_list[i-1]
    
    #Create the gradients
    gradient_list = np.gradient(arclen_list)

    #Plot for gradient:
    #print_gradient_list(gradient_list)

    #Do the nonlinear approximation of the change of arclenth over points.
    L = 650
    epsilon = 95
    C, phis, xl = nonlinear_approximation(reduc_pca, gradient_list, epsilon, L, [])
    grad_approx = np.matmul(phis, C)

    #Visualize the real gradients vs approximation
    #grad_approx_vis(gradient_list, grad_approx, L, epsilon)

    real_len = np.cumsum(gradient_list)
    approx_len = np.cumsum(grad_approx)
    print("The Arclength is: ", arclen_list[-1])
    print("The Archlength according to gradient is: ", real_len[-1])
    print("The Archlength according to approximated gradient is: ",approx_len[-1])

    total_grad_list = []
    for i in range(14):
        C_temp, phis, xl_temp = nonlinear_approximation(reduc_pca, gradient_list, epsilon, L, xl)
        grad_approx = np.matmul(phis, C)
        total_grad_list.append(np.cumsum(grad_approx))##### total grad list is actually cumsummed.

    total_grad_list = np.array(total_grad_list)
    #print(total_grad_list.shape)
    ###Arclength over points for 14 periods:
    
    total_grad_list = total_grad_list.flatten() 
    #plot_arclen_all_periods(total_grad_list)

    df = df[1000:]
    #df = df.iloc[: , c_array]
    data_np = df.to_numpy()
    #data_np = data_np[:13651, 1]
    data_np = data_np[:1997, 1]
    
    #seven_grad_list = total_grad_list[:13651]
    seven_grad_list = total_grad_list[:1997]
    print(seven_grad_list.shape)
    print(data_np.shape)
    L = 650
    epsilon = 95
    C, phis, xl = nonlinear_approximation(seven_grad_list[:, np.newaxis], data_np[:, np.newaxis], epsilon, L, [])
    column_two_predict = np.matmul(phis, C)
    fig, ax = plt.subplots(1, 1)
    ax.plot(column_two_predict, color='indianred', linewidth=5.0, label="Original Data")
    ax.plot(data_np, color='dodgerblue', label="Nonlinear Approximation")
    ax.set_title('L = {}, $\epsilon$ = {:.3f}'.format(L, epsilon))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()

    print(total_grad_list.shape)
    all_column_two_pred = []
    number_of_rows = seven_grad_list.shape[0]
    for i in range(14):
        phis = np.zeros((number_of_rows, L))
        for l in range(L):
            phis[:, l] = np.exp(-(np.linalg.norm(seven_grad_list[:, np.newaxis] - xl[l], axis=1) ** 2) / epsilon ** 2)
        column_two_predict = np.matmul(phis, C)
        all_column_two_pred.append(column_two_predict)
    

    predictions = np.array(all_column_two_pred)
    predictions = predictions.flatten()
    print(predictions.shape)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(predictions, color='indianred', label="14 day prediction")
    #ax.plot(data_np, color='dodgerblue', label="Nonlinear Approximation")
    ax.set_title('L = {}, $\epsilon$ = {:.3f}'.format(L, epsilon))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()


    data_np = df.to_numpy()
    #data_np = data_np[:13651, 1]
    data_np = data_np[:13561, 1]
    fig, ax = plt.subplots(1, 1)
    ax.plot(predictions[:13561], color='indianred', linewidth=5.0, label="Original Data")
    ax.plot(data_np, color='dodgerblue', label="Nonlinear Approximation")
    ax.set_title('L = {}, $\epsilon$ = {:.3f}'.format(L, epsilon))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.legend()
    plt.show()



    



    






main()