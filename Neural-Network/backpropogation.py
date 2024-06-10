import numpy as np
from simple_classification_neural_network import NeuralNetwork

def backpropogation(self, x, y, w, depth , nodes):
    # for each layer do forward propogation and generate cost
    # start from end layer
    output = NeuralNetwork(lambda_=0.05, learning_rate=0.01).forward_propagation(x, w, depth)
    #a = output[-1]
    m = len(x[1])
    delta = output[-1]-y
    for i in range(depth,0,-1):
        Delta = np.zeros((nodes[i],nodes[i-1]))
        if i == depth:
            delta = output[i]-y
            #print(f"delta for layer {i}: {delta}")
        else:
            first_component = np.matmul(np.transpose(w[i]),delta)
            second_component = np.matmul(np.transpose(output[i]),1-output[i])
            delta = np.matmul(first_component,second_component)
        Delta = Delta + np.matmul(delta,np.transpose(output[i-1]))
        D = (1/m)*Delta + self.lambda_*w[i-1]
        w[i-1] = w[i-1] - self.learning_rate*D
        print(f"delta for layer {i}: {delta}")
    
    return w