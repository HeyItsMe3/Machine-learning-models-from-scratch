import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def weights():
    w1 = np.random.rand(30,5)
    w2  = np.random.rand(5,3)
    return (w1,w2)
def forward_propagation(x, w1,w2):
    z1 = x.dot(w1) # (3,30)*(30,5) = (3,5)
    a1 = sigmoid(z1)

    z2 = a1.dot(w2) # (3,5)*(5,3) = (3,3)
    a2 = sigmoid(z2)

    return a2

def loss(y, out):
    loss = np.square(y - out)
    return np.sum(loss)/len(y)

def back_propagation(x,y,w1,w2,learning_rate=0.1):
    z1 = x.dot(w1) # (3,30)*(30,5) = (3,5)
    a1 = sigmoid(z1) # (3,5)

    z2 = a1.dot(w2) # (3,5)*(5,3) = (3,3)
    a2 = sigmoid(z2) # (3,3)

    d2 = a2 - y # (3,3)
    d1 = d2.dot(w2.T) * a1 * (1-a1) # [(3,3)*(3,5)]*(3,5)*(3,5) = (3,5)

    w1_grad = x.T.dot(d1) # (30,3)*(3,5) = (30,5)
    w2_grad = a1.T.dot(d2) # (5,3)*(3,3) = (5,3)

    w1 = w1 - learning_rate*w1_grad
    w2 = w2 - learning_rate*w2_grad

    return (w1, w2)

def train(x, y, w1,w2, learning_rate=0.01, epochs=1000):
    losses = []
    acc = []
    for i in range(epochs):
        l = []
        for j in range(len(x)):
            print(f"weights before backprop: {[w1,w2]}")
            out = forward_propagation(x[j], w1, w2)
            l.append(loss(y[j], out))
            w1,w2 = back_propagation(x[j], y[j], w1,w2, learning_rate)
            print(f"weights after backprop: {[w1,w2]}")
            if j==3:
                break
        print(f"epoch: {i}, loss: {np.mean(l)}, accuracy: {(1-np.mean(l))*100}")
        acc.append((1-np.mean(l))*100)
        losses.append(np.mean(l))
    return w1, w2, losses, acc

def predict(x, w1,w2):
    Out = forward_propagation(x, w1,w2)
    maxm = 0
    k = 0
    for i in range(len(Out[0])):
        if(maxm<Out[0][i]):
            maxm = Out[0][i]
            k = i
    if(k == 0):
        print("Image is of letter A.")
    elif(k == 1):
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")
    plt.imshow(x.reshape(5, 6))
    plt.show()    



def main():
    # Creating data set
 
    # A
    a =[0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1]
    # B
    b =[0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0]
    # C
    c =[0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0]
    
    # Creating labels
    y =[[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]

    x = [np.array(a).reshape(1,30), np.array(b).reshape(1,30), np.array(c).reshape(1,30)]
    y = np.array(y).reshape(3,3)
    w1,w2 = weights()
    print(f"initial weights: {[w1,w2]}")
    w1, w2, losses, acc = train(x, y, w1,w2, epochs=1, learning_rate=0.3)
    
main()