import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from feed_forward_nn_sigmoidal import SigmoidalFeedForwardNeuralNetwork
from  feed_forward_nn_sigmoidal import SigmoidalFeedForwardNeuralNetworkMetricsApproach
from feed_forward_nn_relu import ReLuFeedForwardNeuralNetwork


def one_hot_y(y):
    one_hot_y = []
    for i in range(len(y)):
        z = np.zeros(np.max(y))
        y_val = np.squeeze(y[i])
        if y_val!=0:
            z[np.squeeze(y[i])-1] = 1
        one_hot_y.append(z)
    return one_hot_y


def run():
    # load data from .npy file
    data = np.load('Neural-Network/data.npy', allow_pickle=True)
    #print(data)

    # Access the variables in the .npy file
    x = data.item().get('X') # (5000, 400)
    y = data.item().get('y')

    y = one_hot_y(y) # (5000, 10)
    
    nodes = [400,25,10]
    learning_rate = 0.01
    epoch = 50
    bias = False

    x_train = x[:4900]
    y_train = y[:4900]

    x_val = x[4900:]
    y_val = y[4900:]

      
    #nn = SigmoidalFeedForwardNeuralNetwork(nodes=nodes, learning_rate=learning_rate, epoch=epoch, bias=bias) # Gradient vanishing issue
    #nn = SigmoidalFeedForwardNeuralNetworkMetricsApproach(nodes=nodes, learning_rate=learning_rate, epoch=epoch, bias=bias) # Gradient vanishing issue
    nn = ReLuFeedForwardNeuralNetwork(nodes=nodes, learning_rate=learning_rate, epoch=epoch, bias=bias) # perfect! >91 percent accuracy
    #print(f"initial weight {nn.weights}")
    w, b = nn.train(x_train, y_train, nn.weights, nn.b, epoch, learning_rate)
    #print(w, b)
    print(nn.predict(x_train[0], w, b))
    print(y_train[0])

    print(nn.predict(x_train[100], w, b))
    print(y_train[100])

    print(nn.predict(x_val[50], w, b))
    print(y_val[50])

    print(nn.predict(x_val[80], w, b))
    print(y_val[80])

    

#run()

def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01

# Initialize weights
conv1_filters = initialize_weights((2, 3, 3))  # 2 filters, 3x3 size
fc1_weights = initialize_weights((2*3*3, 64))  # Assuming input image is 8x8, after conv + pool, it's 3x3, 64 neurons
fc1_biases = np.zeros((64, 1))
fc2_weights = initialize_weights((64, 32))    # 32 neurons in the second hidden layer
fc2_biases = np.zeros((32, 1))
fc3_weights = initialize_weights((32, 10))    # 10 output neurons for 10 classes
fc3_biases = np.zeros((10, 1))

# Example input and label
input_image = np.random.randn(8, 8)
true_label = np.zeros((10, 1))
true_label[3] = 1  # Suppose the correct class is 3

def convolution_function(x, f):
    return np.sum(x*f)
    
def kernel(x):
    return np.array([convolution_activation(convolution_function(x, conv1_filters[i])) for i in range(conv1_filters.shape[0])])

def convolution_activation(x):
    return np.maximum(x, 0)

# Forward Pass
# Convolution
conv1_output = np.array(np.correlate(input_image, conv1_filters[i], mode='valid') for i in range(conv1_filters.shape[0]))
#conv1_output = kernel(input_image)
print(np.correlate(input_image, conv1_filters[i], mode='valid') for i in range(conv1_filters.shape[0]))
