import numpy as np
import cv2
import os
import gc
from time import time

file_dir_1 = '/home/tom/disk_1/world_models_data/random_rollouts/'
file_dir_2 = '/home/tom/disk_1/world_models_data/better_rollouts/'

out_dir = '/home/tom/disk_1/world_models_data/all_rollouts_resized/'


def resize_batch(data):
    
    new_data = np.zeros((100,1000,64,64,3))
    for (i,j) in zip(range(100), range(1000)):
        new_data[i,j] = cv2.resize(data[i,j], dsize=(64,64))
    
    return new_data
    

(_,_,files_1) = os.walk(file_dir_1).__next__()
(_,_,files_2) = os.walk(file_dir_2).__next__()

for file in files_1:
    print('Handling file ', file)
    data = np.load(os.path.join(file_dir_1, file), allow_pickle=True)
    acs = data[:,0,:]
    new_acs = np.zeros((100,1000,3))
    for (i,j) in zip(range(100),range(1000)):
        new_acs[i,j] = acs[i,j]
    del acs
    gc.collect()
    
    obs = data[:,1,:]
    new_obs = np.zeros((100,1000,96,96,3))
    for (i,j) in zip(range(100), range(1000)):
        new_obs[i,j] = obs[i,j]
    del obs
    gc.collect()

    # resize images
    new_obs = resize_batch(new_obs)

    # save
    file_name = os.path.join(out_dir, 'rollouts_resized_{}.npz'.format(str(int(time()))))
    np.savez_compressed(file_name, obs=new_obs, acs=new_acs)

for file in files_2:
    print('Handling file ', file)
    data = np.load(os.path.join(file_dir_2, file), allow_pickle=True)
    acs = data[:,0,:]
    new_acs = np.zeros((100,1000,3))
    for (i,j) in zip(range(100),range(1000)):
        new_acs[i,j] = acs[i,j]
    del acs
    gc.collect()
    
    obs = data[:,1,:]
    new_obs = np.zeros((100,1000,96,96,3))
    for (i,j) in zip(range(100), range(1000)):
        new_obs[i,j] = obs[i,j]
    del obs
    gc.collect()

    # resize images
    new_obs = resize_batch(new_obs)

    # save
    file_name = os.path.join(out_dir, 'rollouts_resized_{}.npz'.format(str(int(time()))))
    np.savez_compressed(file_name, obs=new_obs, acs=new_acs)