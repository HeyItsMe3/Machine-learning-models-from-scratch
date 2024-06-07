import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, depth=3, hidden_layers = 2, output_layer = 1, learning_rate = 0.01, epochs = 1000, batch_size = 32, seed = 0, lambda_ = 0.5, nodes = [3,4,4,1]):
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
    
    def feed_forward(self, x, weights):
        a = x
        for weight in weights:
            a = self.sigmoid_function(weight,a)
        return a
    
    def forward_propagation(self, x, w, depth): # add weights here as parameter
        a = x
        total_a = [x]
        for i in range(depth):
            if i < depth-1:
                a = self.sigmoid_function(w[i],a)
                total_a.append(a)
            else:
                a = self.sigmoid_function(w[i],a)
                total_a.append(a)
        print(f"all layer's activation output: {total_a}")
        return total_a

    
    def cost_function(self, x, y, weights):
        prediction = self.forward_propagation(x, self.depth, self.nodes)
        m = len(x[1])
        ones_h = np.ones(len(prediction))
        ones_y = np.ones(len(y))
        c1 = np.matmul(np.transpose(y),np.log(prediction))
        c2 = np.matmul(np.transpose(ones_y-y), np.log(ones_h-prediction))
        for weight in weights:
            regularization = regularization + (self.lambda_/2)*np.matmul(weight, np.transpose(weight))
        cost = -(1/m)*(c1+c2+regularization)
        return cost
    
    def backpropogation(self, x, y, w, depth , nodes):
        # for each layer do forward propogation and generate cost
        # start from end layer
        output = self.forward_propagation(x, w, depth)
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

model = NeuralNetwork()
x = np.array([[1,1,1],[4,5,6],[7,8,9]])
layers = 3
#model.forward_propagation(x,[np.ones((4,3)),np.ones((4,4)),np.ones((1,4))],depth=3)
result = model.backpropogation(x, np.array([1,0,1]), [np.ones((4,3)),np.ones((4,4)),np.ones((1,4))], 3, [3,4,4,1])
print(result)
