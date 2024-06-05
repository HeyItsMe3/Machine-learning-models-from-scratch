import matplotlib.pyplot as plt
import numpy as np
from linear_regression import LinearRegression

weights = [1,2]
x = [[1,1,1,1,1,1,1],[1,2,3,4,5,6,7]]
iteration = 900
y = [2,5,8,11,14,17,20]
learning_rate = 0.05

lr = LinearRegression()
all_weights = lr.gradient_descent_all_weights_recursive(weights, x, y, iteration, learning_rate, all_weights=[])
iteration = []
cost = []
weight1 = []
weight2 = []
for i in range(len(all_weights)):
    cost.append(lr.cost_function(all_weights[i], x, y))
    iteration.append(i)
    #print(f"weights: {weights[i]} and cost: {lr.cost_function(weights[i], x, y)}")

# Plot1: Cost with iteration
plot = plt.plot(iteration, cost)
plt.show()

#Plot2: cost function with weights
# Generate data for the cost function graph
weight1_range = np.linspace(-100, 100, 100)
weight2_range = np.linspace(-100, 100, 100)
cost_values = np.zeros((len(weight1_range), len(weight2_range)))

# Calculate cost for each combination of weights
for i, w1 in enumerate(weight1_range):
    for j, w2 in enumerate(weight2_range):
        weights = [w1, w2]
        cost_values[i, j] = lr.cost_function(weights, x, y)

# Create 3D plot
weight1_grid, weight2_grid = np.meshgrid(weight1_range, weight2_range)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(weight1_grid, weight2_grid, cost_values, cmap='viridis')

# Set labels and show the plot
ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Cost')
plt.show()


# Plot4: Linear function animation each iteration

import matplotlib.animation as animation

fig, ax = plt.subplots()

# Scatter plot of original data
scatter = ax.scatter(x[1], y, label='Actual')

# Line plot for prediction
line, = ax.plot(x[1], lr.predict(all_weights[0], x))

def update(i):
    line.set_ydata(lr.predict(all_weights[i], x))
    return line,

def init():
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(all_weights), init_func=init, blit=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

plt.show()
