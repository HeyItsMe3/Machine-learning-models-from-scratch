import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, depth, nodes, lambda_, epochs=1000, batch_size=32, seed=0):
        self.depth = depth
        self.lambda_ = lambda_
        self.nodes = nodes
        self.weights = []
        self.learning_rate = 0.001
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.initialize_weights()

    # input weight will be a list of numpy arrays like [np.array([1,2,3]), np.array([4,5,6])] 
    # where weight from layer n will look like [1,2,3], first weight is bias
    def initialize_weights(self):
        for i in range(self.depth):
            self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
                
    def sigmoid_function(self, weights, x):
        return 1/(1+np.exp(-np.matmul(weights,x)))
    
    def feed_forward(self, x, weights):
        a = x # input layer
        for weight in weights:
            a = self.sigmoid_function(weight,a) # apply sigmoid function
            a = np.insert(a,0,np.ones(len(a[0]),axis=0)) # add bias
        # return output layer
        return a[:1]
    
    def forward_propagation(self, x, w, depth): # add weights here as parameter
        a = x # input layer
        total_a = [x] # store all layers' output
        for i in range(depth):
            # for each layer apply sigmoid function and add bias
            if i < depth-1:
                a = self.sigmoid_function(w[i],a) # apply sigmoid function
                a = np.insert(a, 0, np.ones(len(a[0])), axis=0) # add bias
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
        delta = output[-1]-y # calculate delta for last layer

        for i in range(depth,0,-1):
            Delta = np.zeros((nodes[i],nodes[i-1]))
            if i == depth:
                delta = output[i]-y # calculate delta for last layer
                #print(f"delta for layer {i}: {delta}")
            else:
                delta = np.matmul(delta,np.transpose(w[i]))*(output[i]*(1-output[i])) # calculate delta for other layers
            
            # calculate Delta for each layer
            Delta = Delta + np.matmul(delta,np.transpose(output[i-1])) 
            D = (1/m)*Delta
            # update gradient
            D = D + self.lambda_*w[i-1]
            # update weights
            w[i-1] = w[i-1] - self.learning_rate*D
            #print(f"delta for layer {i}: {delta}")
                 
        return w

    def train(self, x, y):
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
    
        return self.weights
    
    def plot_decision_boundary(self, x, y): 
        x_min, x_max = x[0].min() - 1, x[0].max() + 1
        y_min, y_max = x[1].min() - 1, x[1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = self.feed_forward(np.c_[xx.ravel(), yy.ravel()].T, self.weights)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x[0], x[1], c=y, alpha=0.8)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Neural Network Decision Boundary')
        plt.show()

    def predict(self, x):
        return self.feed_forward(x, self.weights)
    
    def evaluate(self, x, y):
        prediction = self.predict(x)
        return np.mean(np.argmax(prediction, axis=0) == np.argmax(y, axis=0))
    
    def test(self, x, y):
        return self.evaluate(x, y)
    
    def save(self, filename):
        np.save(filename, self.weights)

    def load(self, filename):
        self.weights = np.load(filename)
        return self.weights
    
    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights
    

nn = NeuralNetwork(2, [2,2,1], 0.01)
""" x_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0],
                   [0.1, 0.8], [0.2, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4],
                   [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0],
                   [0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5],
                   [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0],
                   [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
x = np.transpose(x_train)
x = np.array([np.ones(29),x[0],x[1]])
y_train =  np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]) """
#model.forward_propagation(x,[np.ones((4,3)),np.ones((4,4)),np.ones((1,4))],depth=3)
#w = [np.ones((4,3)),np.ones((2,4)),np.ones((1,2))]
""" w_0 = np.random.randn(4, 3)/np.sqrt(3)
w_1 = np.random.randn(2, 4)/np.sqrt(4)
w_2 = np.random.randn(1, 2)/np.sqrt(3)
w = [w_0,w_1,w_2] """

x = [[1,1,1,1],[0,0,1,1],[0,1,0,1]]
y_train = [0,0,0,1]

#w = [np.random.randn(2,3),np.random.randn(1,2)]
#print(w)

