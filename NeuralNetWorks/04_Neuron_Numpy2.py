#Neuron thats layered using Numpy
# Here there is no need of single Neuron 
# Multiple weights with row of single inputs

import numpy as np

""" def simpleNeuron(inputs, weights,bias):
       
    nOutput=np.dot(weights,inputs) + bias
    return nOutput """
def layeredNeuron(inputs,weights,bias):
    nOutput=np.dot(weights,inputs)+bias
    return nOutput


inputs = [1.0,2.0,3.0,4.0]
weights = [[0.2,0.8,-0.5,1.0],
            [0.2,0.8,-0.5,1.0],
            [0.2,0.8,-0.5,1.0]
        ]   
bias = [2,1.0,-0.8]


print(layeredNeuron(inputs, weights,bias))