import numpy as np
from costfunction import cost_function
from linear_regression import LinearRegression


def gradient_descent(weights, x, iteration, y, learning_rate):
    total_x = len(x[1])
    prediction = LinearRegression().hypothesis(weights,x)
    if iteration == 0:
        return weights
    else:
        weights = weights - (learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x))
        print(f"weights: {weights} and Iteration: {iteration} and cost: {cost_function(weights, x, y)}")
        return gradient_descent(weights, x, iteration-1, y, learning_rate)
    
#print(gradient_descent(weights=[1,2], x=[[1,1,1,1,1,1,1],[1,2,3,4,5,6,7]], iteration=900, y=[2,5,8,11,14,17,20], learning_rate=0.05))
#weights = gradient_descent(weights=[1,2], x=[[1,1,1],[1,2,3]], iteration=450, y=[3,6,9], learning_rate=0.1)
#print(cost_function(weights, x=[[1,1,1],[1,2,3]], y=[3,6,9]))