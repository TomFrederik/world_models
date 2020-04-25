import numpy as np
import os

data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'
(_,_,files) = os.walk(data_dir).__next__()

files = [data_dir + file for file in files]

data = np.load(files[0], allow_pickle = True)

print(data)
print(data.shape) # (100, 2, 1000)
print(data[:,1].shape) # (100, 1000) , each element is a picture of (96,96,3), 100 runs á 1000 frames
print(data[:,0].shape)  # (100, 1000) , each element is a picture of (96,96,3), 100 runs á 1000 frames
print(data[0,1,0].shape) #(96,96,3)
print(data[0,0,0].shape) #(3,), each element is a 3-tuple that represents the action at that frame
#print(data[0,0,0])
#flat_data = data[:,1].flatten()
#print(flat_data[0])
#print(flat_data[0].shape)
#new_flat = np.array([item for item in flat_data])
#print(new_flat.shape)

