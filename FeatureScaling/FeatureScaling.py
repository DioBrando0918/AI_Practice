import numpy as np
import matplotlib.pyplot as plt

"""
Absolute Maximum Scaling
"""
x = np.arange(0, 20, 0.4)
max_val = np.max(x)
y1 = x/max_val  # 對矩陣中每個元素取正弦
y2 = 50 * np.cos(x)

plt.figure(1)
plt.plot(x, y1, color='red')
plt.plot(x, y2, color="blue")
plt.suptitle("Absolute Maximum Scaling")

"""
Min Max Scaling
"""
plt.figure(2)
y3 = (x - min(x)) / (max(x) - min(x))
plt.plot(x, y2, color="blue")
plt.plot(x, y3, color="red")
plt.title("Min Max Scaling")
plt.show()

"""
Mean Normalization
"""
plt.figure(3)
y4 =
