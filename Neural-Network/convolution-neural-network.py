import numpy as np

class ConvolutionNeuralNetwork:
    def __init__(self, nodes, learning_rate, epoch, bias):
        self.nodes = nodes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.bias = bias
        self.weights = []
        self.b = []
        self.z = []
        self.a

    def train(self, x, y, weights, b, epoch, learning_rate):
        self.weights = weights
        self.b = b
        for i in range(epoch):
            for j in range(len(x)):
                self.forward_propagation(x[j])
                self.backward_propagation(y[j])
        return self.weights, self.b
    
    def forward_propagation(self, x):
        self.z = []
        self.a = []
        for i in range(len(self.nodes)-1):
            if i == 0:
                z = np.dot(x, self.weights[i])
            else:
                z = np.dot(self.a[i-1], self.weights[i])
            self.z.append(z)
            a = self.relu(z)
            self.a.append(a)
        return self.a[-1]
    
    def backward_propagation(self, y):
        for i in range(len(self.nodes)-1, 0, -1):
            if i == len(self.nodes)-1:
                d = self.a[i-1] - y
            else:
                d = np.dot(self.weights[i+1], d)
            self.weights[i] -= self.learning_rate * np.dot(d, self.a[i-1].T)
            self.b[i] -= self.learning_rate * d
        return self.weights, self.b
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def predict(self, x, weights, b):
        self.weights = weights
        self.b = b
        return self.forward_propagation(x)
    
    def accuracy(self, x, y, weights, b):
        self.weights = weights
        self.b = b
        correct = 0
        for i in range(len(x)):
            if np.argmax(self.predict(x[i], self.weights, self.b)) == np.argmax(y[i]):
                correct += 1
        return correct/len(x)
    
    def one_hot_y(self, y):
        one_hot_y = []
        for i in range(len(y)):
            z = np.zeros(np.max(y))
            y_val = np.squeeze(y[i])
            if y_val!=0:
                z[np.squeeze(y[i])-1] = 1
            one_hot_y.append(z)
        return one_hot_y
    
    def run(self):
        # load data from .npy file
        data = np.load('Neural-Network/data.npy', allow_pickle=True)
        #print(data)
    
        # Access the variables in the .npy file
        x = data.item().get('X')
        y = data.item().get('y')

        y = self.one_hot_y(y)

        x_train = x[:4900]
        y_train = y[:4900]

        x_val = x[4900:]
        y_val = y[4900:]

        w, b = self.train(x_train, y_train, self.weights, self.b, self.epoch, self.learning_rate)
        print(self.predict(x_train[0], w, b))
        print(y_train[0])

        print(self.predict(x_train[100], w, b))
        print(y_train[100])

        print(self.predict(x_val[50], w, b))
        print(y_val[50])

        print(self.predict(x_val[80], w, b))
        print(y_val[80])

        print(self.accuracy(x_train, y_train, w, b))
        print(self.accuracy(x_val, y_val, w, b))

cnn = ConvolutionNeuralNetwork(nodes=[400,25,10], learning_rate=0.01, epoch=50, bias=False)
cnn.run()
