# 
# 
from nnfs.datasets import spiral_data
import numpy as np

import nnfs 
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100,classes=3)
plt.scatter(X[:,0], X[:,1],c=y)
plt.show()