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
class NeuralNetworkReLuSoftmax:
    def __init__(self, depth, nodes, lambda_, bias = True, epochs=100, learning_rate=0.001):
        self.depth = depth
        self.lambda_ = lambda_
        self.nodes = nodes
        self.weights = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(self.depth):
            # Xavier initialization (usually used for tanh and sigmoid activation functions)
            #self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1) * np.sqrt(1 / self.nodes[i]))
            # He initialization (usually used for ReLU activation function)
            #self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
            # Random initialization
            #self.weights.append(np.ones((self.nodes[i+1], self.nodes[i]+1)))
            # constant initialization
            self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1) * 0.01)
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def ReLu(self, z):
        return np.maximum(0,z)
             
    def z_function(self, weights, x):
        return np.matmul(weights,x)
    
    def Relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    def feed_forward(self, x, weights):
        a = x # input layer
        for i in range(len(weights)):
            if i < len(weights)-1:
                z = self.z_function(weights[i],a)
                a = self.ReLu(z)
                if self.bias:
                    bias_random = np.random.rand(len(a[0]))
                    a = np.insert(a, 0, bias_random, axis=0)
                else:
                    zero_bias = np.zeros(len(a[0]))
                    a = np.insert(a, 0, zero_bias, axis=0)
            else:
                z = self.z_function(weights[i],a)
                a = self.softmax(z)
        return a
    
    def forward_propagation(self, x, w, depth): # add weights here as parameter
        a = x # input layer
        total_a = [x] # store all layers' output
        for i in range(depth):
            # for each layer apply sigmoid function and add bias
            if i == 0:
                z = self.z_function(w[i],a)
                a = self.ReLu(z)
                if self.bias:
                    bias_random = np.random.rand(len(a[0]))
                    a = np.insert(a, 0, bias_random, axis=0) # add bias
                else:
                    zero_bias = np.zeros(len(a[0]))
                    a = np.insert(a, 0, zero_bias, axis=0)
                total_a.append(a)
            elif i < depth-1:
                z = self.z_function(w[i],a)
                a = self.ReLu(z)
                if self.bias:
                    bias_random = np.random.rand(len(a[0]))
                    a = np.insert(a, 0, bias_random, axis=0) # add bias
                else:
                    zero_bias = np.zeros(len(a[0]))
                    a = np.insert(a, 0, zero_bias, axis=0)
                total_a.append(a)
            else:
                z = self.z_function(w[i],a)
                a = self.softmax(z)
                total_a.append(a)
        #print(f"all layer's output: {total_a}")
        return total_a

    
    def backpropagation(self, x, y, w, depth , nodes):
        # for each layer do forward propogation
        # start from end layer
        output = self.forward_propagation(x, w, depth)
        m = len(y) # number of training examples
        gradient =[]
        for i in range(depth,0,-1):
            Delta = np.zeros((nodes[i],nodes[i-1]))
            if i == depth:
                delta = output[i]-y # calculate delta for last layer
                #print(f"delta for layer {i}: {delta}")
            else:
                delta = np.matmul(np.transpose(w[i]), delta) * (self.Relu_derivative(output[i]))

                # remove bias from delta
                delta = delta[1:]

            
            # calculate Delta for each layer
            Delta = np.matmul(delta,np.transpose(output[i-1])) 
            D = (1/m)*Delta
            # update gradient
            #D = D + (self.lambda_/m)*w[i-1]
            # store gradient
            gradient.append(D)
        
        # update weights
        gradient = gradient[::-1]
                 
        return gradient
    
        
    def train(self, x, y):
        # for each epoch do backpropagation and update weights
        w = self.weights
        for epoch in range(self.epochs):
            # update weights
            #print(f"weights: {w} in epoch {epoch}")
            gradient = self.backpropagation(x, y, w, self.depth, self.nodes)
            #print(f"gradient: {gradient}")
            for i in range(self.depth-1,-1,-1):
                w[i] = w[i] - self.learning_rate*gradient[i] 

                # for every 50 iterations check accuracy
            if epoch % 50 == 0:
               print(f"accuracy: {np.mean(self.predict(x, w) == np.argmax(y, axis=0))}")

            #print(f"weights: {w}")
            #print(f"epoch: {epoch}")
            #print(f"batch: {i}")
    
        return self.weights


    def predict(self, x, w):
        p = self.feed_forward(x, w)
        return np.argmax(p, axis=0)
    
    def store_weight(self, x, y):
        w = self.train(x,y)
        np.save("Neural-Network/weight.npy", w)

def main():
    nn = NeuralNetworkReLuSoftmax(2, [400,25,10], 0, bias = True, epochs=1000, learning_rate=0.1)
    data = np.load('Neural-Network/data.npy', allow_pickle=True)
    X = data.item().get('X')
    y = data.item().get('y')
    x = X.T
    x = np.insert(x, 0, np.ones(len(x[0])), axis=0)
    y = y.T

    w = nn.weights
    print(f"weights: {w}") 

    #train
    w = nn.train(x, y)

    # get weights after training
    """ weight = np.load('Neural-Network/weight.npy', allow_pickle=True)
    w = weight.item.get('w')
    x = x[:, 6]
    x_test = np.reshape(x,(len(x),1))
    print(f"predicted output: {nn.predict(x_test, w)}")
    
    first_column = x[:, 6]
    first_column = first_column.T
    image = first_column.reshape((20, 20))*255
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show() """
main()