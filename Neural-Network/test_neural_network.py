import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from simple_sigmoidal_neural_network_m import NeuralNetwork




""" print(x.shape)
print(y.shape)
print(y) """
def run():
    # load data from .npy file
    data = np.load('Neural-Network/data.npy', allow_pickle=True)
    #print(data)

    # Access the variables in the .npy file
    X = data.item().get('X')
    y = data.item().get('y')
    x = X.T
    x = np.insert(x, 0, np.ones(len(x[0])), axis=0)

    #print(x)
    #print(y)
    y = y.T
    nn = NeuralNetwork(2, [400,25,1], 0, epochs=20, learning_rate=0.001, batch_size=0, seed=0) 
    w,cost = nn.train(x, y)
    #print(f"predicted output: {nn.predict(x)}")
    p = nn.predict(x)
    #print(nn.compute_numerical_gradient(x, y, w, nn.depth, nn.nodes))
    #print(nn.check_gradients(x, y))
    # draw a plot between cost and epochs
    print(f'Training Set Accuracy: {np.mean(p == y.flatten()) * 100}%')
    epch = np.arange(nn.epochs,0,-1)
    plt.plot(epch,cost)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs Epochs')
    plt.show()

run()

