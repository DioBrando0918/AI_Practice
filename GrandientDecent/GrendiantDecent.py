import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from mpl_toolkits.mplot3d import Axes3D


def ComputeCost(x, y, theta):
    m = len(x)
    prediction = x.dot(theta)
    square_error = (prediction - y) ** 2
    return 1 / (2 * m) * np.sum(square_error)


def GrandientDescent(x, y, theta, alpha, num_inters):
    m = len(x)
    J_history = []

    for i in range(num_inters):
        predictions = x.dot(theta)
        error = np.dot(x.transpose(), (predictions - y))
        descent = alpha * 1 / m * error
        theta -= descent
        J_history.append(ComputeCost(x, y, theta))
    return theta, J_history


def main():
    data = pd.read_csv('Uni_linear.txt', header=None)
    data_information = data.describe()  # Display data information
    plt.figure(1)
    plt.subplot(1, 1, 1)
    # plt.plot(data[0],data[1])
    plt.scatter(data[0], data[1])
    plt.xticks(np.arange(5, 30, step=5))
    plt.yticks(np.arange(-5, 30, step=5))
    plt.xlabel("Population of City (10,000s)")
    plt.ylabel("Profit ($10,000")
    plt.title("Profit Vs Population")
    data_np = data.values  # convert dataset to np.array
    m = len(data_np[:])
    X = np.append(np.ones((m, 1)), data_np[:, 0].reshape(m, 1), axis=1)
    Y = data_np[:, 1].reshape(m, 1)
    theta = np.zeros((2, 1))  # theta1 = theta0 = 0
    theta, J_history = GrandientDescent(X, Y, theta, 0.01, 1500)
    print(theta)
    print(J_history)
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = ComputeCost(X, Y, t)

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(236, projection='3d')  # row x cloumn:2 x 3 ,locate: 6 -> a26
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap="coolwarm")  # plot_surface only for 3d axies
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("$\Theta_0$")
    ax.set_ylabel("$\Theta_1$")
    ax.set_zlabel("$J(\Theta)$")

    plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.plot(np.arange(1500), J_history)
    plt.xlabel("iterations")
    plt.ylabel("J(Θ)")

    plt.figure(4)
    plt.scatter(data[0],data[1])
    x_value = [i for i in range(25)]
    y_value = [j * theta[1] + theta[0] for j in x_value]
    plt.plot(x_value, y_value, color='red')
    plt.xticks(np.arange(0, 30, step=5))
    plt.yticks(np.arange(0, 30, step=5))
    plt.xlabel("Population of city")
    plt.ylabel("Profit($10000)")
    plt.title("Profit vs Population")
    plt.show()


if __name__ == '__main__':
    main()
