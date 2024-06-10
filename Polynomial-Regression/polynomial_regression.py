import numpy as np

class PolynomialRegression:
    def __init__(self):
        pass

    def hypothesis(self, weights, x):
        function = np.matmul(weights, x)
        return function

    def cost_function(self, weights, x, y):
        total_x = len(x[1])
        prediction = self.hypothesis(weights, x)
        error_func = prediction-y
        cost = (1/(2*total_x))*np.matmul(error_func, error_func)
        return cost
    
    def gradient_descent(self, weights, x, y, iteration, learning_rate):
        total_x = len(x[1])
        prediction = self.hypothesis(weights, x)
        weights = weights-((learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x)))
        if iteration==0:
            return weights
        else:
            print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent(weights, x, y, iteration-1, learning_rate)
        
    def gradient_descent_all_weights(self, weights, x, y, iteration, learning_rate, all_weights = []):
        total_x = len(x[1])
        prediction = self.hypothesis(weights, x)
        weights = weights-((learning_rate/total_x)*np.matmul((prediction-y),np.transpose(x)))
        if iteration==0:
            return all_weights
        else:
            all_weights.append(weights)
            #print(f"weights: {weights} and Iteration: {iteration} and cost: {self.cost_function(weights, x, y)}")
            return self.gradient_descent_all_weights(weights, x, y, iteration-1, learning_rate, all_weights)

# y = a + bx + cx^2 + dx4 can be written as y = a + bx1 + cx2 + dx4 => provide inputs accordingly       
model = PolynomialRegression()
weights = [1,3,2]
x = [[1,1,1,1,1,1],[0,1,2,4,-1,-2],[0,1,4,16,1,4]]
y = [-1,-3,-13,-57,-7,-21]

#print(model.gradient_descent_all_weights(weights, x, y, iteration=800, learning_rate=0.01, all_weights=[]))
#print(model.gradient_descent(weights, x, y, iteration=800, learning_rate=0.01))

#print(model.hypothesis(weights, x))
