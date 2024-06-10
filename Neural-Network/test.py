import numpy as np
import matplotlib.pyplot as plt
from simple_classification_neural_network import NeuralNetwork
# Test the neural network
X = [
    [1,5.1, 3.5],
    [1,4.9, 3.0],
    [1,4.7, 3.2],
    [1,4.6, 3.1],
    [1,5.0, 3.6],
    [1,5.4, 3.9],
    [1,4.6, 3.4],
    [1,5.0, 3.4],
    [1,4.4, 2.9],
    [1,4.9, 3.1],
    [1,5.4, 3.7],
    [1,4.8, 3.4],
    [1,4.8, 3.0],
    [1,4.3, 3.0],
    [1,5.8, 4.0],
    [1,5.7, 4.4],
    [1,5.4, 3.9],

    # Add more data points as needed
]

y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]])

""" 
    [1,5.1, 3.5],
    [1,5.7, 3.8],
    [1,5.1, 3.8]

2, 2, 2
 """

x = np.transpose(X)
nn = NeuralNetwork(3, [2,10,10,1], 0, epochs=2000, learning_rate=0.8, batch_size=4, seed=0) 
w,cost = nn.train(x, y)
x_test = [[1,5.1, 3.5],
    [1,5.7, 3.8],
    [1,5.1, 3.8]]
x_test = np.transpose(x_test)
print(f"predicted output: {nn.predict(x_test)}")
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
