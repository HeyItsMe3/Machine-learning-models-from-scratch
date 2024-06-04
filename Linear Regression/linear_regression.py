import numpy as np

class LinearRegression:
    def __init__(self):
        pass

    def hypothesis(self, weights, x):
        return np.matmul(weights,x)
    
    def cost_function(self, weights, x, y):
        prediction = LinearRegression().hypothesis(weights, x)
        total_x = len(x)
        error_func = prediction-y
        cost = (1/(2*total_x))*np.matmul(error_func, error_func)
        return cost
    
    def gradient_descent(self, weights, x, y, iteration, learning_rate):
        total_x = len(x[1])
        for _ in range(iteration):
            prediction = self.hypothesis(weights,x)
            weights = weights - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
            print(f"weights: {weights} and Iteration: {_} and cost: {self.cost_function(weights, x, y)}")
        
        return weights
        
    def gradient_descent_recursive(self, weights, x, y, iteration, learning_rate):
        total_x = len(x[1])
        prediction = self.hypothesis(weights,x)
        weights = weights - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
        if iteration == 0:
            return weights
        else:
            print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent_recursive(weights, x, y, iteration-1, learning_rate)
        
    def gradient_descent_all_weights_recursive(self, weights, x, y,  iteration, learning_rate, all_weights=[]):
        total_x = len(x[1])
        prediction = self.hypothesis(weights, x)
        weights = weights - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
        if iteration == 0:
            return all_weights
        else:
            all_weights.append(weights)
            #print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent_all_weights_recursive(weights, x, y, iteration-1, learning_rate, all_weights)
    
    def train(self, x_train, y_train):
        weights = np.ones(len(x_train))
        w = self.gradient_descent(weights,x_train,y_train,iteration=100,learning_rate=0.01)
        return w
    
    def predict(self, weights, x_test):
        prediction = self.hypothesis(weights,x_test)
        return prediction