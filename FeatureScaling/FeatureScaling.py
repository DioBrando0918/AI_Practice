import numpy as np
import matplotlib.pyplot as plt

"""
Absolute Maximum Scaling
"""
x = np.arange(0, 20, 0.4)
y1 = np.sin(x)  # 對矩陣中每個元素取正弦
y2 = 50 * np.cos(x)

plt.figure(1)
plt.plot(x, y1, color='red')
plt.plot(x, y2, color="blue")
plt.suptitle("Absolute Maximum Scaling")

"""
Absolute Maximum Scaling
"""
plt.figure(2)
y1_new = (x - min(x)) / (max(x) - min(x))
plt.plot(x, y1_new, color="blue")
plt.plot(x, y2, color="red")
plt.title("Absolute Maximum Scaling")
plt.show()
