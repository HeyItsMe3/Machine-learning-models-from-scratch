import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, depth=4, hidden_layers = 2, output_layer = 1, learning_rate = 0.01, epochs = 1000, batch_size = 32, seed = 0, lambda_ = 0.5, nodes = [3,4,4,1]):
        self.depth = depth
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.nodes = nodes

    # input weight will be a list of numpy arrays like [np.array([1,2,3]), np.array([4,5,6])] 
    # where weight from layer n will look like [1,2,3], first weight is bias
    def sigmoid_function(self, weights, x):
        return 1/(1+np.exp(-np.matmul(weights,x)))
    
    def forward_propgation(self, x, depth, nodes):
        #total_hidder_layers = layers-1
        a = x
        for i in range(depth-1):
            if i < depth-2:
                w = np.ones((nodes[i],nodes[i+1])) # user xavier_uniform_ to initialize weights
                a = self.sigmoid_function(w,a)
                #print(a)
            else:
                w = np.ones((nodes[i],nodes[i+1]))
                a = self.sigmoid_function(w,a) # final output
                #print(a)
        return a
    
    def cost_function(self, x, y, weights):
        prediction = self.forward_propgation(x, self.depth, self.nodes)
        m = len(x[1])
        ones_h = np.ones(len(prediction))
        ones_y = np.ones(len(y))
        c1 = np.matmul(np.transpose(y),np.log(prediction))
        c2 = np.matmul(np.transpose(ones_y-y), np.log(ones_h-prediction))
        for weight in weights:
            regularization = regularization + (self.lambda_/2)*np.matmul(weight, np.transpose(weight))
        cost = -(1/m)*(c1+c2+regularization)
        return cost
    
    
model = NeuralNetwork()
x = np.array([[1,1,1],[4,5,6],[7,8,9]])
layers = 3
model.forward_propgation(x,layers)
w = np.ones((3,3))

