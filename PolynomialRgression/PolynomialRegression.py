import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

plt.figure()
dataset = pd.read_csv("data.csv")
x = np.array(dataset["Level"]).reshape(len(dataset["Level"]), 1)
y = np.array(dataset["Salary"])
plt.scatter(x, y)
plt.xlabel("Level")
plt.xticks(np.arange(dataset["Level"][0], (dataset["Level"][len(dataset["Level"]) - 1]) + 1, step=1))
plt.ylabel("Salary")
plt.title("Position vs Salary")
reg = linear_model.LinearRegression()
reg.fit(x, y)
plt.plot(x, reg.predict(x), color="red")

plt.figure(1)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)  # row1:係數 row2:一次項 row:二次項
reg.fit(x_poly, y)
plt.plot(x_poly, reg.predict(x_poly), color='green')
plt.show()
