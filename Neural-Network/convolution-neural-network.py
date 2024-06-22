import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



class ConvolutionNeuralNetwork:
    def __init__(self, strides, kernals):
        self.strides = strides
        self.kernals = kernals

    def convolution_function(self, x, f):
        return np.sum(x*f)
    
    def kernel(self, x):
        return np.array([self.convolution_function(x, f) for f in self.kernals])

    def convolution_layer(self, x):
        f = len(self.kernals[0])
        n = len(x)
        m = len(x[0])
        k = len(self.kernals)
        new_x = np.zeros((n-f+1, m-f+1, k))
        for i in range(n-f+1):
            for j in range(m-f+1):
                new_x[i,j] = self.kernel(x[i:i+f, j:j+f])
        return new_x
    
def main():
    x = np.array([[0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]])
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
    cnn = ConvolutionNeuralNetwork(strides, kernals)
    # Getting the output image
    x = cnn.convolution_layer(x)


    #print(x[0])
    # Plotting the output image
    plt.imshow(x[0], cmap='gray')
    plt.show()
    plt.imshow(x[1], cmap='gray')
    plt.show()
    # Open the image file
    #img = Image.open('Neural-Network/bird.jpg')
    #print(img)

if __name__ == "__main__":
    main()