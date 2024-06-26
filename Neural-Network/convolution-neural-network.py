import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time



class FeatureExtraction:
    def __init__(self, strides, kernals, bias=True):
        self.strides = strides
        self.kernals = kernals
        self.bias = bias
        self.b = []

    def padding(self, x):
        return np.pad(x, 1, mode='constant', constant_values=0)
    
    def convolution_function(self, x, f):
        return np.sum(x*f)
    
    def kernel(self, x):
        return np.array([self.convolution_function(x, f) for f in self.kernals])

    def convolution_activation(self, x):
        return np.maximum(x, 0)
    
    def convolution_layer(self, x):
        f = len(self.kernals[0])
        n = len(x)
        m = len(x[0])
        k = len(self.kernals)
        new_x = np.zeros((k, n-f+1, m-f+1)) # Depth, Rows, Columns
        #print(f"new_x : {new_x}")
        for k in range(k):
            for i in range(n-f+1):
                for j in range(m-f+1) :
                    new_x[k,i,j] = self.convolution_function(x[i:i+f, j:j+f], self.kernals[k])

        # After activation function
        b = np.random.rand(1, k)*0.01
        if self.bias:
            new_x = self.convolution_activation(new_x + b)
        else:
            new_x = self.convolution_activation(new_x)
        return new_x
    
    def max_pooling(self, x): # Here x is 2d array
        a = len(x)
        b = len(x[0])
        #x = np.array(x)
        new_x = np.zeros((a, b))
        for i in range(0, a):
            for j in range(0, b):
                new_x[i, j] = np.max(x[i:i+2, j:j+2])
        return new_x
    
    def pooling_layer(self, x):
        d = []
        for i in range(len(x)):
            d.append(self.max_pooling(x[i])) 
        return d
    
    def flatten(self, x):
        x = np.array(x)
        return x.flatten() 


class Classification:
    def __init__(self, nodes, learning_rate, epoch, bias=True):
        self.bias = bias
        self.b = []
        self.depth = len(nodes)-1
        self.nodes = nodes
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = []
        self.weight_init()
        self.bias_term()

    def weight_init(self):
        for i in range(self.depth):
            self.weights.append(np.random.rand(self.nodes[i+1], self.nodes[i])*0.01)
        return self.weights
    
    def bias_term(self):
        for i in range(self.depth):
            if self.bias:
                self.b.append(np.ones((1, self.nodes[i+1])))
            else:
                self.b.append(np.zeros((1, self.nodes[i+1])))
        return self.b

    def z_function(self, x, w, b):
        return np.dot(x,w.T) + b
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return z > 0
    
    def sigmoid_activation_function(self, z):
        return (1/(1+np.exp(-z)))
    
    def feed_forward(self, x, w, b):
        a = x
        for i in range(self.depth):
            if i != self.depth-1:
                z = self.z_function(a, w[i], b[i])
                a = self.relu(z)
            else:
                z = self.z_function(a, w[i],b[i])
                a = self.sigmoid_activation_function(z) #self.softmax(z)

        return a
    
    def forward_propagation(self, x, w, b):
        w = self.weights
        b = self.b
        a = x
        all_layer_activation = [x]
        for i in range(self.depth):
            if i != self.depth-1:
                z = self.z_function(a, w[i], b[i])
                a = self.relu(z)
            else:
                z = self.z_function(a, w[i], b[i])
                a = self.sigmoid_activation_function(z) #self.softmax(z)
                
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
                delta = np.dot(delta,w[i]) * self.relu_derivative(output[i])
            
            grad_weight = delta.T * output[i-1]

            if self.bias:
                grad_bias = np.mean(delta, axis=0)
            else:
                grad_bias = 0

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

    

    
def one_hot_y(y):
    one_hot_y = []
    for i in range(len(y)):
        z = np.zeros(np.max(y))
        y_val = np.squeeze(y[i])
        if y_val!=0:
            z[np.squeeze(y[i])-1] = 1
        one_hot_y.append(z)
    return one_hot_y

    
def main():
    start = time.time()

    """ x = np.array([[0, 0, 1, 1, 1, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0],
                  [1, 0, 1, 1, 1, 0, 1],
                  [1, 0, 1, 1, 1, 0, 1],
                  [1, 0, 0, 0, 0, 0, 1],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0]]) """
    
    data = np.load('Neural-Network/data.npy', allow_pickle=True)

    x = data.item().get('X')
    y = data.item().get('y')
    y = one_hot_y(y)

    # train data
    x_train = x[:4000]
    y_train = y[:4000]

    # test data
    x_test = x[4000:]
    y_test = y[4000:]

    # Normalize the data
    x_train = x_train/255
    # resize x into a 3d array
    x_train = np.array([np.array(x_train[i].reshape(20, 20)) for i in range(len(x_train))])
    
    
    # Defining the strides
    strides = 1
    # Defining the kernals
    kernals = np.array([[[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]],
                        [[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]]])
    
    # Creating the object of the ConvolutionNeuralNetwork class
    feature_extraction = FeatureExtraction(strides, kernals, bias = False)

    # convolution layey/Feature extraction
    nn_input = []
    for i in range(len(x_train)): 
        x_padded = feature_extraction.padding(x_train[i])
        convoluted_output = feature_extraction.convolution_layer(x_padded)
        output_after_pooling = feature_extraction.pooling_layer(convoluted_output)
        input_vector = feature_extraction.flatten(output_after_pooling)
        nn_input.append(input_vector)
    #print(np.shape(nn_input[0]))
    
    nodes = [len(nn_input[0]), 20, 10]
    learning_rate = 0.01
    epoch = 200
    bias = False
    classification = Classification(nodes, learning_rate, epoch, bias)

    w, b = classification.train(nn_input, y_train, classification.weights, classification.b, epoch, learning_rate)
    print(classification.predict(nn_input[0], w, b))
    print(y_train[0])

    print(classification.predict(nn_input[400], w, b))
    print(y_train[400])

    print(classification.predict(nn_input[2000], w, b))
    print(y_train[2000])

    print(classification.predict(nn_input[3000], w, b))
    print(y_train[3000])

    # test data
    x_test = x_test/255
    x_test = np.array([np.array(x_test[i].reshape(20, 20)) for i in range(len(x_test))])
    nn_input_test = []
    for i in range(len(x_test)): 
        x_padded = feature_extraction.padding(x_test[i])
        convoluted_output = feature_extraction.convolution_layer(x_padded)
        output_after_pooling = feature_extraction.pooling_layer(convoluted_output)
        input_vector = feature_extraction.flatten(output_after_pooling)
        nn_input_test.append(input_vector)
    print(classification.predict(nn_input_test[100], w, b))
    print(y_test[100])

    print(classification.predict(nn_input_test[400], w, b))
    print(y_test[400])

    print(classification.predict(nn_input_test[800], w, b))
    print(y_test[800])



    

    
    # Plotting the output image
    """ plt.imshow(x[0], cmap='gray')
    plt.show()
    plt.imshow(x[1], cmap='gray')
    plt.show() """

    end = time.time()

    print(f"Time taken: {end-start}")
    
if __name__ == "__main__":
    main()