import numpy as np
import matplotlib.pyplot as plt

seed = 10

np.random.seed(10)
a = np.random.normal(0,1,5)
np.random.seed(10)
b = np.random.normal(0,2,5)

plt.plot(a/b)
plt.show()