#Forward Pass - with hidden layers (layered Neurans)
# np.dot (Inputs, transpose(Weights))
#output1 = np.dot (Inputs1, Transpose(Weights1)) + bias1
#output2 = np.dot (Inputs1, Transpose(Weights2)) + bias2 //Hidden layer1.. similarly, there can be N hidden layers

import numpy as np

""" def simpleNeuron(inputs, weights,bias):
       
    nOutput=np.dot(weights,inputs) + bias
    return nOutput """
def layeredNeuron(inputs,weights,bias):
    nOutput=np.dot(inputs, np.array(weights).T)+bias
    return nOutput

def forwardPassNeuralNetwork(inputs,weights_hiddenLayers,bias,noOfLayers):
    interation=0
    layeredOutput =any
    while (interation < noOfLayers):
        if noOfLayers == 0:
            layeredOutput = layeredNeuron(inputs,weights_hiddenLayers[interation],bias[interation])
            
        else:
            layeredOutput = layeredNeuron(layeredOutput,weights_hiddenLayers[interation],bias[interation])
        interation+=1
    return layeredOutput





inputs = [[1.0,2.0,3.0,4.0],
          [1.2,-0.8,0.4,5]]
#Hidden layers
weights_hiddenLayers = [[[0.2,0.8,-0.5,1.0],[0.2,0.8,-0.5,1.0],[0.2,0.8,-0.5,1.0]],
                        [[0.2,0.8,-0.5,1.0],[0.2,0.8,-0.5,1.0],[0.2,0.8,-0.5,1.0]],
                        [[0.2,0.8,-0.5,1.0],[0.2,0.8,-0.5,1.0],[0.2,0.8,-0.5,1.0]]]  
bias = [[2,1.0,-0.8],
        [2,1.0,-0.8],
        [2,1.0,-0.8] ]
noOfLayers = 3

print(forwardPassNeuralNetwork(inputs,weights_hiddenLayers,bias,noOfLayers))