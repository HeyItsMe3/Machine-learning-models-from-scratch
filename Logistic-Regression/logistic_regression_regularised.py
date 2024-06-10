import numpy as np

class LogisticRegressionRegularised:
    def __init__(self):
        pass

    def sigmoid_function(self, weights, x):
        return 1/(1+np.exp(-np.matmul(weights,x)))
    
    def cost_function(self, weights, x, y, lambda_):
        total_x = len(x[1])
        prediction = self.sigmoid_function(weights, x)
        ones_y = np.ones(len(y))
        ones_h = np.ones(len(prediction))
        c1 = np.matmul(np.transpose(y),np.log(prediction))
        regularization = (lambda_/2)*np.matmul(weights,np.transpose(weights))
        c2 = np.matmul(np.transpose(ones_y-y),np.log(ones_h-prediction))
        cost = -(1/total_x)*(c1+c2+regularization)
        return cost
    
    
    def gradient_descent(self, weights, x, y, iteration, learning_rate, lambda_):
        total_x = len(x[1])
        for _ in range(iteration):
            prediction = self.sigmoid_function(weights,x)
            weights = weights*(1-((learning_rate*lambda_)/total_x)) - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
            print(f"weights: {weights} and Iteration: {_} and cost: {self.cost_function(weights, x, y, lambda_)}")
        
        return weights
    
    def gradient_descent_all_Weights(self, weights, x, y, iteration, learning_rate, lambda_, all_weights=[]):
        total_x = len(x[1])
        
        for _ in range(iteration):
            prediction = self.sigmoid_function(weights,x)
            weights = weights*(1-((learning_rate*lambda_)/total_x)) - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
            all_weights.append(weights)
            print(f"weights: {weights} and Iteration: {_} and cost: {self.cost_function(weights, x, y, lambda_)}")

        return all_weights
    
    def train(self, x_train, y_train, lambda_=0.5, weights=np.array([]), iteration=100, learning_rate=0.01):
        w = self.gradient_descent(weights,x_train,y_train,iteration=iteration,learning_rate=learning_rate, lambda_=lambda_)
        return w
    
    def predict(self, weights, x_test):
        prediction = self.sigmoid_function(weights,x_test)
        return prediction
    