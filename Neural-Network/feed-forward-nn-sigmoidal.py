import numpy as np
import matplotlib.pyplot as plt

"""
Parameters and Dimensions

    x => (3,30) features => 30,  training example => 3
    y => (3,3) labels => 3, training example =>  3
    nodes => [30, 3, 2, 3] => input layer => 30, hidden-layers => 3 and 2, output-layer => 3
    Depth => 3
    weights => (3,30), (2,3), (3,2)
"""
class SigmoidalFeedForwardNeuralNetwork():
    def __init__(self, nodes, learning_rate, epoch, bias = True):
        self.depth = len(nodes)-1
        self.nodes = nodes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.bias = bias
        self.b = []
        self.weights = []
        self.weight_init()
        self.bias_term()

    def weight_init(self):
        for i in range(self.depth):
            self.weights.append(np.random.rand(self.nodes[i+1], self.nodes[i]))
        return self.weights
    
    def bias_term(self):
        for i in range(self.depth):
            self.b.append(np.ones((1, self.nodes[i+1])))
        return self.b

    def z_function(self, x, w, b):
        return np.dot(x,w.T) + b
    
    def sigmoid_activation_function(self, z):
        return (1/(1+np.exp(-z)))
    
    def feed_forward(self, x, w, b):
        a = x
        for i in range(self.depth):
            z = self.z_function(a, w[i], b[i])
            a = self.sigmoid_activation_function(z)

        return a
    
    def forward_propagation(self, x, w, b):
        w = self.weights
        b = self.b
        a = x
        all_layer_activation = [x]
        for i in range(self.depth):
            z = self.z_function(a, w[i], b[i])
            a = self.sigmoid_activation_function(z)
            all_layer_activation.append(a)
        
        return all_layer_activation
    
    def back_propagation(self, x, y, w, b, learning_rate):
        output = self.forward_propagation(x, w, b)
        dw = []
        db = []
        for i in range(self.depth, 0, -1):
            if i == self.depth:
                delta = output[i] - y
            else:
                delta = np.dot(delta,w[i]) * (output[i] * (1-output[i]))
            
            grad_weight = delta.T * output[i-1]
            grad_bias = np.mean(delta, axis=0)

            dw.append(grad_weight)
            db.append(grad_bias)

        # update weights
        for j in range(len(w), 0, -1):
            w[j-1] = w[j-1] - learning_rate*dw[-j]
            b[j-1] = b[j-1] - learning_rate*db[-j]

        return w, b
    
    def train(self, x, y, w, b, epoch, learning_rate):
        m = len(y)
        for e in range(epoch):
            for i in range(m):
                #print(f"Weights before backpropagation: {w} bias: {b}")
                w, b = self.back_propagation(x[i], y[i], w, b, learning_rate)
            output = self.forward_propagation(x, w, b)
            cost = np.mean(np.square(output[-1]-y))
            acc = (1-cost)*100
            print(f"epoch: {e} cost: {cost} accuracy: {acc}")
        
        return w, b

    def predict(self, x, w, b):
        prediction = self.feed_forward(x, w, b)
        return prediction
    
    
def main():
    # Creating data set
 
    # A
    a =[0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1]
    # B
    b =[0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0]
    # C
    c =[0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0]
    
    # Creating labels
    y =[[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]

    x = [a,b,c]
    y = np.array(y).reshape(3,3)
    nn = SigmoidalFeedForwardNeuralNetwork(nodes=[30,10,3], learning_rate=0.1, epoch=100, bias=True)

    w, b = nn.train(x, y, nn.weights, nn.b, epoch=500, learning_rate=0.3)
    print(nn.predict(x[0], w, b))
    print(nn.predict(x[1], w, b))
    print(nn.predict(x[2], w, b))

    
main()


