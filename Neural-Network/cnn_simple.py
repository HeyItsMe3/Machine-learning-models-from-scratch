import numpy as np

"""
The structure of the CNN:

1. **Convolutional Layer (conv1)**
2. **ReLU Activation**
3. **Pooling Layer (pool1)**
4. **Flattening Layer**
5. **Fully Connected Layer (fc1)**
6. **ReLU Activation (hidden layer 1)**
7. **Fully Connected Layer (fc2)**
8. **ReLU Activation (hidden layer 2)**
9. **Output Layer (fc3) with Softmax**

Here’s the updated code with two hidden fully connected layers:

"""

# Helper functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def cross_entropy_loss(pred, true):
    return -np.sum(true * np.log(pred + 1e-15))  # Adding epsilon for numerical stability

def cross_entropy_loss_derivative(pred, true):
    return pred - true

def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01

# Initialize weights
conv1_filters = initialize_weights((2, 3, 3))  # 2 filters, 3x3 size
fc1_weights = initialize_weights((2*3*3, 64))  # Assuming input image is 8x8, after conv + pool, it's 3x3, 64 neurons
fc1_biases = np.zeros((64, 1))
fc2_weights = initialize_weights((64, 32))    # 32 neurons in the second hidden layer
fc2_biases = np.zeros((32, 1))
fc3_weights = initialize_weights((32, 10))    # 10 output neurons for 10 classes
fc3_biases = np.zeros((10, 1))

# Example input and label
input_image = np.random.randn(8, 8)
true_label = np.zeros((10, 1))
true_label[3] = 1  # Suppose the correct class is 3

def convolution_function(x, f):
    return np.sum(x*f)
    
def kernel(x):
    return np.array([convolution_activation(convolution_function(x, conv1_filters[i])) for i in conv1_filters.shape[0]])

def convolution_activation(x):
    return np.maximum(x, 0)

# Forward Pass
# Convolution
conv1_output = np.array([relu(np.correlate(input_image, conv1_filters[i], mode='valid')) for i in range(conv1_filters.shape[0])])

# Pooling (2x2, max pool)
pool1_output = np.array([conv1_output[i, ::2, ::2] for i in range(conv1_output.shape[0])])

# Flatten
flattened_output = pool1_output.flatten().reshape(-1, 1)

# Fully Connected Layer 1
fc1_z = np.dot(fc1_weights.T, flattened_output) + fc1_biases
fc1_a = relu(fc1_z)

# Fully Connected Layer 2
fc2_z = np.dot(fc2_weights.T, fc1_a) + fc2_biases
fc2_a = relu(fc2_z)

# Output Layer
fc3_z = np.dot(fc3_weights.T, fc2_a) + fc3_biases
output_z = softmax(fc3_z)

# Loss
loss = cross_entropy_loss(output_z, true_label)

# Backward Pass
# Output Layer
dL_dz3 = cross_entropy_loss_derivative(output_z, true_label)

# Fully Connected Layer 2
dL_da2 = np.dot(fc3_weights, dL_dz3)
dL_dz2 = dL_da2 * relu_derivative(fc2_z)
dL_dw2 = np.dot(fc1_a, dL_dz2.T)
dL_db2 = dL_dz2

# Fully Connected Layer 1
dL_da1 = np.dot(fc2_weights, dL_dz2)
dL_dz1 = dL_da1 * relu_derivative(fc1_z)
dL_dw1 = np.dot(flattened_output, dL_dz1.T)
dL_db1 = dL_dz1

# Update Fully Connected Layer weights and biases
learning_rate = 0.01
fc3_weights -= learning_rate * np.dot(fc2_a, dL_dz3.T)
fc3_biases -= learning_rate * dL_dz3

fc2_weights -= learning_rate * dL_dw2
fc2_biases -= learning_rate * dL_db2

fc1_weights -= learning_rate * dL_dw1
fc1_biases -= learning_rate * dL_db1

# Flatten Layer
dL_dflat = np.dot(fc1_weights, dL_dz1)

# Reshape back to pooled layer shape
dL_dpool = dL_dflat.reshape(pool1_output.shape)

# Pooling Layer
dL_dconv = np.zeros_like(conv1_output)
for i in range(dL_dpool.shape[0]):
    for j in range(pool1_output.shape[1]):
        for k in range(pool1_output.shape[2]):
            patch = conv1_output[i, j*2:j*2+2, k*2:k*2+2]
            (max_j, max_k) = np.unravel_index(np.argmax(patch), patch.shape)
            dL_dconv[i, j*2 + max_j, k*2 + max_k] = dL_dpool[i, j, k]

# Convolutional Layer
dL_dconv = dL_dconv * relu_derivative(conv1_output)
dL_dfilter = np.zeros_like(conv1_filters)
for i in range(conv1_filters.shape[0]):
    for j in range(conv1_filters.shape[1]):
        dL_dfilter[i, j] = np.correlate(input_image, dL_dconv[i], mode='valid')[j:j+1]

# Update Convolutional Layer filters
conv1_filters -= learning_rate * dL_dfilter

print(f"Updated conv1_filters: {conv1_filters}")
print(f"Updated fc1_weights: {fc1_weights}")
print(f"Updated fc1_biases: {fc1_biases}")
print(f"Updated fc2_weights: {fc2_weights}")
print(f"Updated fc2_biases: {fc2_biases}")
print(f"Updated fc3_weights: {fc3_weights}")
print(f"Updated fc3_biases: {fc3_biases}")

"""
### Key Points:

1. **Additional Fully Connected Layers**:
   - Added `fc2` with corresponding weights (`fc2_weights`), biases (`fc2_biases`), and ReLU activation.

2. **Forward Pass**:
   - Included the second fully connected layer.

3. **Backward Pass**:
   - Computed the gradients for the second hidden layer (`fc2`).
   - Updated the weights and biases for both fully connected layers (`fc1`, `fc2`, and `fc3`).

4. **Gradient Calculation**:
   - Used the chain rule to backpropagate errors from the output layer to the input layer, updating weights and biases at each step.

"""