import numpy as np

data = np.load('./data/random_rollouts_0.npy', allow_pickle = True)

print(data)
print(data.shape)
print(data[:,1].shape)
print(data[:,0].shape)
print(data[0,1].shape)
print(data[0,0].shape)

