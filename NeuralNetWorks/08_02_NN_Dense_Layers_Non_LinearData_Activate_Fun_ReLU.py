#Dense layer class -> will produce output by randomly assigning weights to inputs and bias on multiple layers. 
#np.dot (X,trans(W)) + B
# We will activate the output using ReLU activation function.
# 
from nnfs.datasets import spiral_data
import numpy as np
import nnfs 
import matplotlib.pyplot as plt
from matplotlib import interactive

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons) # ramdomly creating weights for now using Guassian distribution.. 
        self.biases = np.zeros((1,n_neurons))
    
    #forward pass
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) +self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Exponential:
    def forward(self,inputs):
        self.output = np.exp(inputs)

#create dataset 
nnfs.init() #initilize the random seed
X,y = spiral_data(samples=100, classes=3)


#create an object 
dense_l1=Layer_Dense(2,3)

#lets implement first layer..
dense_l1.forward(X)
print(dense_l1.output)
plt.plot(dense_l1.output)
plt.show()

exp=Activation_Exponential()
exp.forward(dense_l1.output)

plt.plot(exp.output)
plt.show()

relu=Activation_ReLU()
relu.forward(dense_l1.output)

plt.plot(relu.output)
plt.show()

dense_l2=Layer_Dense(3,3)
dense_l2.forward(dense_l1.output)
plt.plot(dense_l2.output)
plt.show()

M,n = spiral_data(samples=100, classes=3)
print(M)
plt.plot(M)


plt.show()



