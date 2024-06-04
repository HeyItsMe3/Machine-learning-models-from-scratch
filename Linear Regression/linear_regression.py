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
    
    def gradient_descent(self, weights, x, iteration, y, learning_rate):
        total_x = len(x[1])
        prediction = self.hypothesis(weights,x)
        weights = weights - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
        if iteration == 0:
            return weights
        else:
            #print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent(weights, x, iteration-1, y, learning_rate)
        
    def gradient_descent_all_weights(self, weights, x, iteration, y, learning_rate, all_weights=[]):
        total_x = len(x[1])
        prediction = self.hypothesis(weights, x)
        weights = weights - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
        if iteration == 0:
            return all_weights
        else:
            all_weights.append(weights)
            #print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent_all_weights(weights, x, iteration-1, y, learning_rate, all_weights)
    

model = LinearRegression()
weights = [1,2]
#x = [[1,1,1],[1,2,3]]
#y = [3,6,9]
#print(model.hypothesis(weights, x))
#print(model.gradient_descent_all_weights(weights=[1,2], x=[[1,1,1,1,1,1,1],[1,2,3,4,5,6,7]], iteration=900, y=[2,5,8,11,14,17,20], learning_rate=0.05, all_weights=[]))
#print(model.gradient_descent(weights=[1,2], x=[[1,1,1,1,1,1,1],[1,2,3,4,5,6,7]], iteration=900, y=[2,5,8,11,14,17,20], learning_rate=0.05))
