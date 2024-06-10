import numpy as np
import scipy.io 
import time


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def debug_initialize_weights(fan_in, fan_out):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(np.arange(1, np.size(W) + 1)), W.shape) / 10
    return W

""" def debug_initialize_weights(L_in, L_out): # Accuracy differs each time with random initialization
    epsilon_init = 0.12
    w = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    w = np.reshape(w, (L_out, 1 + L_in))
    return w """

""" def debug_initialize_weights(L_in, L_out): # Accuracy differes each time with random initialization
    # Xavier initialization
    w = np.random.randn(L_out, 1 + L_in) * np.sqrt(1 / L_in)
    w = np.reshape(w, (L_out, 1 + L_in))
    return w """

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    
    m = X.shape[0]
    J = 0
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)
    
    X = np.column_stack([np.ones(m), X])
    z2 = Theta1.dot(X.T)
    a2 = sigmoid(z2)
    
    a2 = np.row_stack([np.ones(m), a2])
    z3 = Theta2.dot(a2)
    h_theta = sigmoid(z3)
    
    y_matrix = np.zeros((num_labels, m))
    for i in range(m):
        y_matrix[y[i] - 1, i] = 1
    
    J = (1 / m) * np.sum(-y_matrix * np.log(h_theta) - (1 - y_matrix) * np.log(h_theta))
    
    t1 = Theta1[:, 1:]
    t2 = Theta2[:, 1:]
    Reg = (lambda_ / (2 * m)) * (np.sum(np.square(t1)) + np.sum(np.square(t2)))
    
    J = J + Reg
    
    for t in range(m):
        a1 = X[t, :][:, np.newaxis]
        z2 = Theta1.dot(a1)
        a2 = sigmoid(z2)
        
        a2 = np.vstack(([1], a2))
        z3 = Theta2.dot(a2)
        a3 = sigmoid(z3)
        
        delta_3 = a3 - y_matrix[:, t][:, np.newaxis]
        z2 = np.vstack(([1], z2))
        delta_2 = Theta2.T.dot(delta_3) * sigmoid_gradient(z2)
        delta_2 = delta_2[1:]
        
        Theta2_grad = Theta2_grad + delta_3.dot(a2.T)
        Theta1_grad = Theta1_grad + delta_2.dot(a1.T)
    
    Theta2_grad = (1 / m) * Theta2_grad
    Theta1_grad = (1 / m) * Theta1_grad
    
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    
    return J, grad


def optimize_weights(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    
    from scipy.optimize import minimize
    
    result = minimize(nn_cost_function, nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_), method='CG', jac=True, options={'maxiter': 100})
    # cg: conjugate gradient: a method for finding the minimum of a function of many variables
    
    return result.x

def train_neural_network(X, y, input_layer_size, hidden_layer_size, num_labels, lambda_):
    initial_Theta1 = debug_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = debug_initialize_weights(hidden_layer_size, num_labels)
    
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])
    
    nn_params = optimize_weights(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
    
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    
    return Theta1, Theta2

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    
    p = np.zeros(m)
    
    X = np.column_stack([np.ones(m), X])
    z2 = Theta1.dot(X.T)
    a2 = sigmoid(z2)
    
    a2 = np.row_stack([np.ones(m), a2])
    z3 = Theta2.dot(a2)
    h_theta = sigmoid(z3)
    
    p = np.argmax(h_theta, axis=0)
    
    return p + 1

def main():
    start  = time.time()
    mat_data = scipy.io.loadmat('Neural-Network/data1.mat')
    X = mat_data['X']
    y = mat_data['y']
    
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_ = 0.1
    
    Theta1, Theta2 = train_neural_network(X, y, input_layer_size, hidden_layer_size, num_labels, lambda_)
    
    p = predict(Theta1, Theta2, X)
    
    print(f'Training Set Accuracy: {np.mean(p == y.flatten()) * 100}%')

    end = time.time()
    print(f"Time taken: {end-start} seconds")

if __name__ == '__main__':
    main()
