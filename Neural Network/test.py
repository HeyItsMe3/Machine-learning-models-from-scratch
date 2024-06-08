import numpy as np
weights = []
nodes = [2,2,1]
x = np.array([[1,1,1,1],[0,0,1,1],[0,1,0,1]])
def initialize_weights():
    for i in range(2):
        if i == 0:
            weights.append(np.random.randn(nodes[i+1], nodes[i]+1))
        else:
            weights.append(np.random.randn(nodes[i+1], nodes[i]+1))

    #print(f"weights: {weights}")

def sigmoid_function(weights, x):
    return 1/(1+np.exp(-np.matmul(weights,x)))

initialize_weights()

a = x
total_a = [x]
depth = 2

def forward_propagation(x, w, depth): # add weights here as parameter
        a = x
        total_a = [x]
        for i in range(depth):
            if i < depth-1:
                a = sigmoid_function(w[i],a)
                a = np.insert(a, 0, np.ones(len(a[0])), axis=0)
                total_a.append(a)
            else:
                a = sigmoid_function(w[i],a)
                total_a.append(a)
        #print(f"all layer's activation output: {total_a}")
        return total_a

#print(forward_propagation(x,weights,2))

matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[7, 8, 9], [10, 11, 12]])

x = np.array([[1,1,1,1],[0,0,1,1],[0,1,0,1]])
y = np.array([[0,1,1,0]])
print(y.shape)

x_batch = x[:,1:50]
y_batch = y[:,0:30]

print(y_batch)