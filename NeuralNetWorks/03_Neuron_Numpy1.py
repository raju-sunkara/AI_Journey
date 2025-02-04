#Neuron thats layered using Numpy

import numpy as np

def simpleNeuron(inputs, weights,bias):
       
    nOutput=np.dot(weights,inputs) + bias
    return nOutput
def layeredNeuron(inputs,weights,bias):
    nOutput=simpleNeuron(inputs, weights[0],bias[0]) +simpleNeuron(inputs, weights[1],bias[1])+simpleNeuron(inputs, weights[2],bias[2])
    return nOutput


inputs = [1.0,2.0,3.0,4.0]
weights = [[0.2,0.8,-0.5,1.0],
            [0.2,0.8,-0.5,1.0],
            [0.2,0.8,-0.5,1.0]
        ]   
bias = [2,1.0,-0.8]


print(layeredNeuron(inputs, weights,bias))