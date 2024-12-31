#Dense layer class -> will produce output by randomly assigning weights to inputs and bias on multiple layers. 
#np.dot (X,trans(W)) + B
# 
# 

import numpy as np

import nnfs
from nnfs.datasets import spiral_data

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons) # ramdomly creating weights for now using Guassian distribution.. 
        self.biases = np.zeros((1,n_neurons))
    
    #forward pass
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) +self.biases

#create dataset 
X,y = spiral_data(samples=100, classes=3)

#create an object 
dense_l1=Layer_Dense(2,3)

#lets implement first layer..
dense_l1.forward(X)
print(dense_l1.output[:5])

