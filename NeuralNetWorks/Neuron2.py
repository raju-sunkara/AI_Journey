//Neuron
inputs = [1.0,2.0,3.0,4.0]
weights = [0.2,0.8,-0.5,1.0]
bias = 2
nOutput = 0 
for x,y in zip(inputs,weights):
    nOutput=nOutput+x*y
nOutput=nOutput + bias
print(nOutput)