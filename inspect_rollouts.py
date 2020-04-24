import numpy as np

data = np.load('/home/tom/data/random_rollouts_0_500.npy', allow_pickle = True)

#print(data)
#print(data.shape) # (500, 2, 1000)
#print(data[:,1].shape) # (500, 1000) , each element is a picture of (96,96,3)
#print(data[:,0].shape)
#print(data[0,1,0].shape) #(96,96,3)
#print(data[0,0,0].shape) #(3,)
flat_data = data[:,1].flatten()
print(flat_data[0])
print(flat_data[0].shape)
new_flat = np.array([item for item in flat_data])
print(new_flat.shape)

