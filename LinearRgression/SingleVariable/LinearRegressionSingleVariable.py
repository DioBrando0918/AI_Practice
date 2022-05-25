import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Home_prices.csv")
plt.xlabel('area')
plt.ylabel('prices(US$)')
reg = linear_model.LinearRegression()
reg.fit(df[['area']].values, df.price)  # fit need to 2D array
# print(reg.coef_, reg.intercept_)  # theta1,theta0
data = pd.read_csv('areas.csv')
prices = reg.predict(np.array(data['area']).reshape(len(data['area']), 1))
data['prices'] = prices
data.to_csv(index=False)

"""
training set predict
"""
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(np.array(df['area']).reshape(len(df['area']), 1)), color='blue')
plt.show()

"""
new data prdict
"""
plt.scatter(data.area, prices, color='red', marker='+')
plt.plot(data.area, reg.predict(np.array(data['area']).reshape(len(data['area']), 1)), color='blue')
plt.show()
