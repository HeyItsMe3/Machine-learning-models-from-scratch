import numpy as np

# 2 layer neural network
class SimpleNeuralNetwork:
    def __init__(self, depth, nodes, lambda_):
        self.depth = depth
        self.nodes = nodes
        self.lambda_ = lambda_
        self.weights = []
        self.learning_rate = 0.001
        self.epochs = 1000
        self.batch_size = 32
        self.seed = 0
        self.initialize_weights()
    
    def initialize_weights(self):
        for i in range(self.depth):
            if i == 0:
                self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
            else:
                self.weights.append(np.random.randn(self.nodes[i+1], self.nodes[i]+1))
        
    def sigmoid_function(self, weights, x):
        return 1/(1+np.exp(-np.matmul(weights,x)))
    
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
    
    def backpropagation(self, x, y, w, depth, nodes):
        # for each layer do forward propagation and generate cost
        # start from end layer
        output = self.forward_propagation(x, w, depth)
        m = len(x[1])
        delta = output[-1] - y
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
            D = D + self.lambda_ * w[i - 1]
            w[i - 1] = w[i - 1] - self.learning_rate * D
        return w
            
    def train(self, x, y):
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
        return self.weights
    
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

# Test the neural network
x = np.array([[1,1,1,1],[0,0,1,1],[0,1,0,1]])
y = np.array([[0,1,1,0]])
nn = SimpleNeuralNetwork(2, [2,2,1], 0.01) 
nn.train(x, y)
print(nn.predict(x))
print(nn.test(x, y))
#nn.save("weights.npy")
#nn.load("weights.npy")
print(nn.get_weights())
print(nn.test(x, y))
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



