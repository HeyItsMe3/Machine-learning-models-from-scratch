import numpy as np
import matplotlib.pyplot as plt

# 2 layer neural network
class SimpleNeuralNetwork:
    def __init__(self, depth, nodes, lambda_, epochs = 1000, learning_rate = 0.001, batch_size=32, seed=0):
        self.depth = depth
        self.nodes = nodes
        self.lambda_ = lambda_
        self.weights = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.initialize_weights()
    
    def initialize_weights(self):
        for i in range(self.depth):
            # Xavier initialization
            self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1) * np.sqrt(1 / self.nodes[i]))
            
            # He initialization            
            #self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
            
            # Random initialization
            #self.weights.append(np.ones((self.nodes[i+1], self.nodes[i]+1)))
        
    def sigmoid_function(self, weights, x):
        return 1/(1+np.exp(-np.matmul(weights,x)))
    
    def linear_function(self, weights, x):
        return np.matmul(weights,x)
    
    def activation_function(self, x):
        return (1/(1+np.exp(-x)))
    
    def neuron_output(self, weights, x):
        return self.activation_function(self.linear_function(weights, x))
    
    def sigmoid_gradient(self, z):
        return self.activation_function(z) * (1 - self.activation_function(z))
    
    def cost_function(self, x, y, weights):

    def feed_forward(self, x, weights):
        a = x
        for weight in weights:
            a = self.sigmoid_function(weight,a)
            a = np.insert(a, 0, np.ones(len(a[0])), axis=0)
        return a[1:]
    
    def forward_propagation(self, x, w, depth): # add weights here as parameter
        a = x
        total_a = [x]
        for i in range(depth):
            if i < depth-1:
                a = self.sigmoid_function(w[i],a)
                # add bias
                a = np.insert(a, 0, np.ones(len(a[0])), axis=0)
                total_a.append(a)
            else:
                a = self.sigmoid_function(w[i],a)
                total_a.append(a)
        #print(f"all layer's activation output: {total_a}")
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
    
    def backpropagation(self, x, y, w, depth, nodes): # BIG BUG HERE, YOU ARE NOT UPDATING THE WEIGHTS CORRECTLY, for EACH TRAINING EXAMPLE YOU NEED TO UPDATE THE WEIGHTS
        # for each layer do forward propagation and generate cost
        # start from end layer
        output = self.forward_propagation(x, w, depth)
        m = len(x[1])
        #delta = output[-1] - y
        gradient = []
        for i in range(depth, 0, -1):
            Delta = np.zeros((nodes[i], nodes[i - 1]))            
            if i == depth:
                delta = output[i]-y
            else:
                delta = np.matmul(np.transpose(w[i]), delta) * output[i] * (1 - output[i])
                # remove bias from delta
                delta = delta[1:]

            Delta = np.matmul(delta, np.transpose(output[i - 1]))
            D = (1 / m) * Delta
            D = D + (self.lambda_/m) * w[i - 1]
            gradient.append(D)
        gradient = gradient[::-1]
        for i in range(depth-1,-1,-1):
            w[i] = w[i] - self.learning_rate*gradient[i] 
            #w[i - 1] = w[i - 1] - self.learning_rate * D
        return w
            
    def train(self, x, y):
        c = []
        for epoch in range(self.epochs):
            for i in range(0, len(x[1]), self.batch_size):
                x_batch = x[:,i:i+self.batch_size]
                y_batch = y[:,i:i+self.batch_size]
                self.weights = self.backpropagation(x_batch, y_batch, self.weights, self.depth, self.nodes)
                print(f"weights: {self.weights}")
                print(f"cost: {self.cost_function(x_batch, y_batch, self.weights)}")
                print(f"epoch: {epoch}")
                print(f"batch: {i}")
                print(f"")
            c = np.insert(c, 0, self.cost_function(x, y, self.weights), axis=0)    
        return self.weights,c
    
    def predict(self, x):
        return self.feed_forward(x, self.weights)
    
    def evaluate(self, x, y):
        prediction = self.predict(x)
        #print(f"prediction: {prediction}")
        return np.mean(np.argmax(prediction, axis=0) == np.argmax(y, axis=0)) # check it later
    
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

# Test the neural network
x = np.array([[1,1,1,1],[0,0,1,1],[0,1,0,1]])
y = np.array([[0,1,1,0]])
nn = SimpleNeuralNetwork(6, [2,5,5,4,3,2,1], 0, epochs=170, learning_rate=0.1, batch_size=4, seed=0) 
w,cost = nn.train(x, y)
print(f"predicted output: {nn.predict(x)}")
print(f"binary prediction: {(nn.predict(x) > 0.5).astype(int)}")
#print(nn.test(x, y))
#print(cost)

# draw a plot between cost and epochs
epch = np.arange(nn.epochs,0,-1)
plt.plot(epch,cost)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost vs Epochs')
plt.show()
#nn.save("weights.npy")
#nn.load("weights.npy")
#print(nn.get_weights())
#print(nn.test(x, y))
# Expected output:
# weights: [array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]]), array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]])]
# cost: [[0.69314718]]
# epoch: 0
# batch: 0

# weights: [array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]]), array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]])]
# cost: [[0.69314718]]
# epoch: 0
# batch: 32

# weights: [array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]]), array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]])]
# cost: [[0.69314718]]
# epoch: 1
# batch: 0
# optimal weights: [array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]]), array([[ 0.00414153, -0.00423485,  0.00414153],
#        [-0.00423485,  0.00414153, -0.00423485]])]
# [[0.5 0.5 0.5 0.5]]


