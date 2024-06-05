import numpy as np

class LogisticRegression:
    def __init__(self):
        pass

    def sigmoid_function(self, weights, x):
        h = np.matmul(weights,x)
        return 1/(1+np.exp(-h))
    
    def cost_function(self, weights, x, y):
        total_x = len(x[1])
        h = self.sigmoid_function(weights, x)
        c1 = np.matmul(np.transpose(y),np.log(h))
        ones_y = np.ones(len(y))
        ones_h = np.ones(len(h))
        c2 = np.matmul(np.transpose(ones_y-y),np.log(ones_h-h))
        cost = -(1/total_x)*(c1+c2)
        return cost
    
    def gradient_descent(self, weights, x, y, iteration, learning_rate):
        total_x = len(x[1])
        for _ in range(iteration):
            h = self.sigmoid_function(weights,x)
            weights = weights - (learning_rate/total_x)*np.matmul((h-y),np.transpose(x))
            #print(f"weights: {weights} and cost: {self.cost_function(weights, x, y)}")

        return weights
        
    def gradient_descent_recursive(self, weights, x, y, iteration, learning_rate):
        total_x = len(x[1])
        h = self.sigmoid_function(weights,x)
        weights = weights - (learning_rate/total_x)*np.matmul((h-y),np.transpose(x))
        if iteration == 0:
            return weights
        else:
            #print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent_recursive(weights, x, y, iteration-1, learning_rate)

    def train(self, x_train, y_train):
        weights = np.ones(len(x_train))
        w = self.gradient_descent(weights,x_train,y_train,iteration=100,learning_rate=0.01)
        return w
    
    def predict(self, weights, x_test):
        prediction = self.sigmoid_function(weights,x_test)
        return prediction

