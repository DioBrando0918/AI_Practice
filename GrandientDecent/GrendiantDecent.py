import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston


def ComputeCost(x, y, theta1):
    m = len(x)
    prediction = x.dot(theta1)
    square_error = (prediction - y) ** 2
    return (1 / 2 * m) * np.sum(square_error)


def main():
    data = pd.read_csv('Uni_linear.txt', header=None)
    data_information = data.describe()  # Display data information
    plt.scatter(data[0], data[1])
    plt.xticks(np.arange(5, 30, step=5))
    plt.yticks(np.arange(-5, 30, step=5))
    plt.xlabel("Population of City (10,000s)")
    plt.ylabel("Profit ($10,000")
    plt.title("Profit Vs Population")
    data_np = data.values  # convert dataset to np.array
    m = len(data_np[:])
    x = np.append()

if __name__ == '__main__':
    main()
