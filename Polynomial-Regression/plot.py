import matplotlib.pyplot as plt
import numpy as np
from polynomial_regression import PolynomialRegression

weights = [1,3,2]
x = [[1,1,1,1,1,1],[0,1,2,4,-1,-2],[0,1,4,16,1,4]]
y = [-1,-3,-13,-57,-7,-21]
iteration = 800
learning_rate = 0.01

pr = PolynomialRegression()
all_weights = pr.gradient_descent_all_weights(weights, x, y, iteration, learning_rate, all_weights=[])
iteration = []
cost = []
weight1 = []
weight2 = []
weight3 = []
for i in range(len(all_weights)):
    cost.append(pr.cost_function(all_weights[i], x, y))
    iteration.append(i)
    #print(f"weights: {weights[i]} and cost: {pr.cost_function(weights[i], x, y)}")

# Plot1: Cost with iteration
plot = plt.plot(iteration, cost)
plt.show()

