import numpy as np

class LinearRegressionRegularised:
    def __init__(self):
        pass

    def hypothesis(self, weights, x):
        return np.matmul(weights,x)
    
    def cost_function(self, weights, x, y, lambda_):
        prediction = self.hypothesis(weights, x)
        total_x = len(x[1])
        error_func = prediction-y
        cost = (1/(2*total_x))*np.matmul(error_func, error_func) + (lambda_/(2*total_x))*np.matmul(weights,np.transpose(weights))
        return cost
    
    def gradient_descent(self, weights, x, y, iteration, learning_rate, lambda_):
        total_x = len(x[1])
        for _ in range(iteration):
            prediction = self.hypothesis(weights,x)
            weights = weights*(1-((learning_rate*lambda_)/total_x)) - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
            print(f"weights: {weights} and Iteration: {_} and cost: {self.cost_function(weights, x, y, lambda_)}")
        
        return weights
    
    def gradient_descent_all_weights(self,weights, x, y, iteration, learning_rate, lambda_, all_weights=[]):
        total_x = len(x[1])
        
        for _ in range(iteration):
            prediction = self.hypothesis(weights,x)
            weights = weights*(1-((learning_rate*lambda_)/total_x)) - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
            all_weights.append(weights)
            print(f"weights: {weights} and Iteration: {_} and cost: {self.cost_function(weights, x, y, lambda_)}")

        return all_weights
    
    def train(self, x_train, y_train, lambda_=0.5, weights=np.array([]), iteration=100, learning_rate=0.01):
        w = self.gradient_descent(weights,x_train,y_train,iteration=iteration,learning_rate=learning_rate, lambda_=lambda_)
        return w
    
    def predict(self, weights, x_test):
        prediction = self.hypothesis(weights,x_test)
        return prediction
        
