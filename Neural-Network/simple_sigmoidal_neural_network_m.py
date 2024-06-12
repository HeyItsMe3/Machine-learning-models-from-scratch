import numpy as np
import matplotlib.pyplot as plt

"""
this class is a simple implementation of a neural network with backpropagation algorithm
activation function used is sigmoid 
each layer has a bias term
bias is already added in the input data
input are in this format: [[1,1,1,1],[2,4,5,6],[3,7,8,9]], here bias is already added, there are three feeatures and 4 training examples
each column in the input is training example
output is in this format: [[1,0,1,3]], here there are 4 training examples and 1 output classes
"""
class NeuralNetwork:
    def __init__(self, depth, nodes, lambda_, epochs=100, learning_rate=0.001, batch_size=32, seed=0):
        self.depth = depth
        self.lambda_ = lambda_
        self.nodes = nodes
        self.weights = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(self.depth):
            # Xavier initialization (usually used for tanh and sigmoid activation functions)
            self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1) * np.sqrt(1 / self.nodes[i]))
            # He initialization (usually used for ReLU activation function)
            #self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
            # Random initialization
            #self.weights.append(np.ones((self.nodes[i+1], self.nodes[i]+1)))
                
    def sigmoid_function(self, weights, x):
        return 1/(1+np.exp(-np.matmul(weights,x)))
    
    def feed_forward(self, x, weights):
        a = x # input layer
        for weight in weights:
            a = self.sigmoid_function(weight,a) # apply sigmoid function
            # random bias
            bias_ones = np.ones(len(a[0]))
            #bias_random = np.random.randn(len(a[0])) * np.sqrt(1 / len(a[0]))
            a = np.insert(a,0,bias_ones, axis=0) # add bias
        # return output layer
        return a[1:]
    
    def forward_propagation(self, x, w, depth): # add weights here as parameter
        a = x # input layer
        total_a = [x] # store all layers' output
        for i in range(depth):
            # for each layer apply sigmoid function and add bias
            if i < depth-1:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                bias_ones = np.ones(len(a[0]))
                #bias_random = np.random.randn(len(a[0])) * np.sqrt(1 / len(a[0])) # cost is fluctuating to much with random bias
                a = np.insert(a, 0, bias_ones, axis=0) # add bias
                total_a.append(a)
            # for last layer apply sigmoid function
            else:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                total_a.append(a)
        #print(f"all layer's output: {total_a}")
        return total_a

    
    def cost_function(self, x, y, weights):
        prediction = self.forward_propagation(x, weights, self.depth)[-1]
        m = len(x[1])
        ones_h = np.ones(len(prediction))
        ones_y = np.ones(len(y))
        c1 = np.matmul(np.log(prediction),np.transpose(y))
        c2 = np.matmul(np.log(ones_h-prediction),np.transpose(ones_y-y))
        regularization = 0
        for weight in weights:
            for w in weight:
                regularization = regularization + (self.lambda_/2)*np.matmul(w, np.transpose(w))
        cost = -(1/m)*(c1+c2+regularization)
        return cost
    
    def backpropagation(self, x, y, w, depth , nodes):
        # for each layer do forward propogation
        # start from end layer
        output = self.forward_propagation(x, w, depth)
        #print(f"output: {output}")
        m = len(x[1]) # number of training examples
        gradient =[]
        for i in range(depth,0,-1):
            Delta = np.zeros((nodes[i],nodes[i-1]))
            if i == depth:
                delta = output[i]-y # calculate delta for last layer
                #print(f"delta for layer {i}: {delta}")
            else:
                delta = np.matmul(np.transpose(w[i]), delta) * (output[i] * (1-output[i])) # calculate delta for other layers. There is  probably a bug here
                #in the next iteration, delta is with the updated weight in previous layer, is this how it should be??
                # BETTER APPROACH WILL BE TO FIRST CALCULATE ALL THETA GRADIENTS AND THEN UPDATE WEIGHTS`` => YES

                # remove bias from delta
                delta = delta[1:]

            
            # calculate Delta for each layer
            Delta = np.matmul(delta,np.transpose(output[i-1])) 
            D = (1/m)*Delta
            # update gradient
            D = D + (self.lambda_/m)*w[i-1]
            # store gradient
            gradient.append(D)
        
        # update weights
        gradient = gradient[::-1]
        for i in range(depth-1,-1,-1):
            w[i] = w[i] - self.learning_rate*gradient[i] 
                 
        return w
    

    def compute_numerical_gradient(self, x, y, w, depth, nodes):
        epsilon = 1e-5
        numgrad = []
        for i in range(depth):
            numgrad.append(np.zeros((nodes[i+1], nodes[i]+1)))
            for j in range(nodes[i+1]):
                for k in range(nodes[i]+1):
                    w_plus = w.copy()
                    w_minus = w.copy()
                    w_plus[i][j][k] = w_plus[i][j][k] + epsilon
                    w_minus[i][j][k] = w_minus[i][j][k] - epsilon
                    cost_plus = self.cost_function(x, y, w_plus)
                    cost_minus = self.cost_function(x, y, w_minus)
                    numgrad[i][j][k] = (cost_plus - cost_minus) / (2 * epsilon)
        return numgrad
    
    def check_gradients(self, x, y):
        numgrad = self.compute_numerical_gradient(x, y, self.weights, self.depth, self.nodes)
        grad = []
        for i in range(self.depth):
            grad.append(np.zeros((self.nodes[i+1], self.nodes[i]+1)))
            for j in range(self.nodes[i+1]):
                for k in range(self.nodes[i]+1):
                    grad[i][j][k] = self.backpropagation(x, y, self.weights, self.depth, self.nodes)[i][j][k]
        for i in range(self.depth):
            print(f"Layer {i+1}")
            print(f"Numerical Gradient: {numgrad[i]}")
            print(f"Gradient: {grad[i]}")
            print(f"Relative Error: {np.linalg.norm(numgrad[i] - grad[i]) / np.linalg.norm(numgrad[i] + grad[i])}")
            print("")
        
    def train(self, x, y):
        c = []
        # for each epoch do backpropagation and update weights
        for epoch in range(self.epochs):
            for i in range(0, len(x[1]), self.batch_size):
                x_batch = x[:,i:i+self.batch_size] # get batch of training examples
                y_batch = y[:,i:i+self.batch_size] # get batch of labels ?????
                # update weights
                self.weights = self.backpropagation(x_batch, y_batch, self.weights, self.depth, self.nodes)
                print(f"weights: {self.weights}")
                print(f"cost: {self.cost_function(x_batch, y_batch, self.weights)}")
                print(f"epoch: {epoch}")
                print(f"batch: {i}")
                print(f"")
            # store cost for each epoch
            c = np.insert(c, 0, self.cost_function(x, y, self.weights), axis=0)  
    
        return self.weights, c
    
    def train_until_cost_minimum(self, x, y):
        c = []
        # for each epoch do backpropagation and update weights
        #for epoch in range(self.epochs):
        cycle = 0
        while True:
            if self.cost_function(x, y, self.weights) < 0.1:
                break

            self.weights = self.backpropagation(x, y, self.weights, self.depth, self.nodes)
            print(f"weights: {self.weights}")
            print(f"cost: {self.cost_function(x, y, self.weights)}")
            print(f"cycle: {cycle}")
            print(f"")
            # store cost for each cycle
            c = np.insert(c, 0, self.cost_function(x, y, self.weights), axis=0)  

            cycle = cycle + 1
        return self.weights, c, cycle


    def predict(self, x):
        return self.feed_forward(x, self.weights)
    
