import numpy as np
import matplotlib.pyplot as plt

"""

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

    # input weight will be a list of numpy arrays like [np.array([1,2,3]), np.array([4,5,6])] 
    # where weight from layer n will look like [1,2,3], first weight is bias
    def initialize_weights(self):
        for i in range(self.depth):
            # Xavier initialization
            self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1) * np.sqrt(1 / self.nodes[i]))
            # He initialization
            #self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
            # Random initialization
            #self.weights.append(np.ones((self.nodes[i+1], self.nodes[i]+1)))

                       
    def sigmoid_function(self, weights, x): 
        return 1/(1+np.exp(-np.matmul(weights,x.T)))
    
    """
    Arguments:
    x -- input data, it is a vector, ex: [3,4,5]
    weights -- weights of all layers

    Returns:
    a -- output of the last layer

    """
    def feed_forward(self, x, weights):
        a = x # input layer
        for weight in weights:
            a = self.sigmoid_function(weight,a) # apply sigmoid function
            # add bias to the output
            a = np.insert(a, 0, 1)
            #print(f"weights: {weight}, output: {a}")
        return a[1:]
    
    """
    Arguments:
    x -- input data of one training example
    w -- weights of all layers pertaining to one training example
    depth -- number of layers

    Returns:
    total_a -- list of all layers' output for one training example

    """
    def forward_propagation_per_example(self, x, w, depth): # add weights here as parameter
        a = x # input layer
        x = np.reshape(x, (1,len(x))) # x- is a vector, reshape it to 2D array
        total_a = [x] # store all layers' output
        for i in range(depth):
            # for each layer apply sigmoid function and add bias
            if i < depth-1:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                a = np.insert(a, 0, 1) # add bias
                a = np.reshape(a, (1,len(a))) # a is a vector, reshape it to 2D array
                total_a.append(a)
            # for last layer apply sigmoid function
            else:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                a = np.reshape(a, (1,len(a))) # a is a vector, reshape it to 2D array
                total_a.append(a)
        #print(f"all layer's output: {total_a}")
        return total_a
    
    def forward_propagation(self, x, w, depth): # add weights here as parameter
        a = x # input layer
        total_a = [] # store all layers' output
        for i in range(depth):
            # for each layer apply sigmoid function and add bias
            if i < depth-1:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                a = np.insert(a, 0, np.ones(len(x)), axis=1) # add bias
                total_a.append(a)
            # for last layer apply sigmoid function
            else:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                total_a.append(a)
        #print(f"all layer's output: {total_a}")
        return total_a

    """
    Arguments:
    x -- all input data
    y -- all output data
    weights -- weights of all layers

    Returns:
    cost -- cost function value

    """
    
    def cost_function(self, x, y, weights):
        prediction = self.forward_propagation(x, weights, self.depth)[-1]
        m = len(x)
        ones_h = np.ones(len(prediction))
        ones_y = np.ones(len(y))
        c1 = np.matmul(np.log(prediction),y)
        c2 = np.matmul(np.log(ones_h-prediction),(ones_y-y))
        regularization = 0
        for weight in weights:
            for w in weight:
                regularization = regularization + (self.lambda_/2)*np.matmul(w, np.transpose(w))
        cost = -(1/m)*(c1+c2+regularization)
        cost = np.sum(cost)
        return cost
    
    def Delta(self, depth, nodes):
        Delta = []
        # delta zeros array for each layer
        for i in range(depth):
            Delta.append(np.zeros((nodes[i+1], nodes[i]+1)))
        return Delta

    def backpropagation(self, x, y, depth , nodes):
        m = len(x) # number of training examples
        w = self.weights
        Delta = self.Delta(depth, nodes)
        gradient = []
        for j in range(m):
            # for each layer do forward propogation
            # start from end layer
            output = self.forward_propagation_per_example(x[j], w, depth)
            #print(f"output: {output}")
            for i in range(depth,0,-1):
                if i == depth:
                    delta = output[i]-y[j] # calculate delta for last layer
                    #print(f"delta for layer {i}: {delta}")
                else:
                    delta = np.matmul(np.transpose(w[i]), delta) * (output[i] * (1-output[i])) # calculate delta for other layers. There is  probably a bug here
                    #in the next iteration, delta is with the updated weight in previous layer, is this how it should be??
                    # BETTER APPROACH WILL BE TO FIRST CALCULATE ALL THETA GRADIENTS AND THEN UPDATE WEIGHTS`` => YES

                # remove bias from delta for layer 1
                    delta = delta[1:]

                
                # calculate Delta for each layer
                Delta[i-1] = Delta[i-1] + delta * output[i-1]

        for i in range(depth):
            D = (1/m)*Delta[i-1]
            D = D + (self.lambda_/m)*w[i-1]
            gradient.append(D)     
        
        """ # update weights
        gradient = gradient[::-1]
        for i in range(depth-1,-1,-1):
            w[i] = w[i] - self.learning_rate*gradient[i]  """
                 
        return gradient[::-1]
        

nn = NeuralNetwork(3, [2, 2, 2, 1], 0.01)
x = np.array([[3,4],[4,5]])
# add bias
x = np.insert(x, 0, np.ones(len(x)), axis=1)
y = np.array([[1],[2]])
w = nn.weights
print(f"weights: {w}")  
print(nn.sigmoid_function(w[0],x[0])) # for each training example, here input should be array of metrics
print(nn.feed_forward(x[0],w)) # for each training example
print(nn.forward_propagation_per_example(x[0],w,3)) # for each training example
print(nn.forward_propagation(x,w,3)) # for all training examples
print(nn.cost_function(x,y,w)) # for all training examples
print(nn.Delta(nn.depth, nn.nodes)) # for all training examples
d = nn.Delta(nn.depth, nn.nodes)
print(nn.backpropagation(x,y,nn.depth,nn.nodes)) # for all training examples

