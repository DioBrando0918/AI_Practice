import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
model = np.poly1d(np.polyfit(x, y, 3))  # polyfit:最小二乘法 deg= 多項式最高次方 , poly1d:定義多項式函數
print(model)
x_smoothing = np.linspace(1, 22, 1000)
plt.scatter(x, y)
plt.plot(x, model(x), color='red')

plt.figure(2)
plt.scatter(x, y)
plt.plot(x_smoothing, model(x_smoothing), color='green')
plt.show()
