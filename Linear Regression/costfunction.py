import numpy as np
from linear_regression import LinearRegression

def cost_function(weights, x, y):
    prediction = LinearRegression().hypothesis(weights, x)
    total_x = len(x)
    error_func = prediction-y
    cost = (1/(2*total_x))*np.matmul(error_func, error_func)
    return cost
    
