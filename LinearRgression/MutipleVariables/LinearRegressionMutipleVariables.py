import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn import linear_model
import math

df = pd.read_csv('HomePrices.csv')
median_bedrooms = math.floor(df.bedrooms.median())  # 取中位數
df.bedrooms = df.bedrooms.fillna(median_bedrooms)  # 用一個特定值取代所有nan
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
predict = reg.predict([[88888000, 3, 2]])
print(predict)
