import numpy as np
import os

data_dir = os.path(__file__) + '/data/'
(_,_,files) = os.walk(data_dir).next()
data = np.load(files[0], allow_pickle = True)

print(data)
print(data.shape) # (500, 2, 1000)
print(data[:,1].shape) # (500, 1000) , each element is a picture of (96,96,3)
print(data[:,0].shape)
print(data[0,1,0].shape) #(96,96,3)
print(data[0,0,0].shape) #(3,)
#flat_data = data[:,1].flatten()
#print(flat_data[0])
#print(flat_data[0].shape)
#new_flat = np.array([item for item in flat_data])
#print(new_flat.shape)

