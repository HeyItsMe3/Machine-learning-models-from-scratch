import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from simple_classification_neural_network import NeuralNetwork


# Load the .mat file
mat_data = scipy.io.loadmat('Neural-Network/data1.mat')
# Access the variables in the .mat file
X = mat_data['X']
y = mat_data['y']
x = X.T
x = np.insert(x, 0, np.array(len(x[0])), axis=0)

""" print(x.shape)
print(y.shape)
print(y) """
def run():
    # Load the .mat file
    mat_data = scipy.io.loadmat('Neural-Network/data1.mat')
    # Access the variables in the .mat file
    X = mat_data['X']
    y = mat_data['y']
    x = X.T
    x = np.insert(x, 0, np.array(len(x[0])), axis=0)
    y = y.T
    nn = NeuralNetwork(2, [400,25,1], 3, epochs=100, learning_rate=0.001, batch_size=2500, seed=0) 
    w,cost = nn.train(x, y)
    #print(f"predicted output: {nn.predict(x)}")
    #print(f"binary prediction: {(nn.predict(x) > 0.5).astype(int)}")
    #print(nn.test(x, y))
    #print(nn.compute_numerical_gradient(x, y, w, nn.depth, nn.nodes))
    #print(nn.check_gradients(x, y))
    # draw a plot between cost and epochs
    epch = np.arange(nn.epochs,0,-1)
    plt.plot(epch,cost)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs Epochs')
    plt.show()
run()


