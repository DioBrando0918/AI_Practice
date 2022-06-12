import numpy as np
import matplotlib.pyplot as plt

"""
Absolute Maximum Scaling 
Range: -1~1
"""
x = np.arange(0, 20, 0.4)
max_val = np.max(x)
y1 = 50 * np.cos(x)
y2 = x / max_val  # 對矩陣中每個元素取正弦

plt.figure(1)
plt.plot(x, y1, color='blue')
plt.plot(x, y2, color="red")
plt.suptitle("Absolute Maximum Scaling")

"""
Min Max Scaling
Range: 0~1
"""
plt.figure(2)
y3 = (y1 - min(y1)) / (max(y1) - min(y1))
plt.plot(x, y1, color="blue")
plt.plot(x, y3, color="red")
plt.title("Min Max Scaling")

"""
Mean Normalization
"""
plt.figure(3)
y4 = (y1 - np.mean(y1)) / (np.max(y1) - np.min(y1))
plt.plot(x, y1, color="blue")
plt.plot(x, y4, color="red")
plt.title("Mean Normalization")
"""
Standardization
"""
plt.figure(4)
y5 = (y1 - np.mean(y1)) / np.std(y1)
plt.plot(x, y1, color="blue")
plt.plot(x, y5, color="red")
plt.title("Standradlization")
plt.show()

"""
Robust Scaling
"""
