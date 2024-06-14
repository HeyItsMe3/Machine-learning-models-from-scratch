import numpy as np 
import matplotlib.pyplot as plt

data = np.load('Neural-Network/data.npy', allow_pickle=True)
    #print(data)



weight = np.load('Neural-Network/weight.npy', allow_pickle=True)
w = weight.item.get('w')
print(weight)

#print(first_column.shape)
