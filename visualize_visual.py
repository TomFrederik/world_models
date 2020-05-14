'''
inspect how well the variational autoencoder works
'''

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# other ptyhon modules
import argparse
from time import time
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# my modules
import modules


def main(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    deconv_layers = config.deconv_layers
    z_dim = config.latent_dim

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    cur_dir = os.path.dirname(os.path.realpath(__file__))

    model_dir = cur_dir + '/models/'
    data_dir = cur_dir + '/data/'

    (_,_,model_files) = os.walk(model_dir).__next__()
    (_,_,data_files) = os.walk(data_dir).__next__()

    # use AE or VAE?
    deterministic = False

    # set up model
    if deterministic:
        encoder = modules.Det_Encoder(input_dim, conv_layers, z_dim)
        decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
        model = modules.AE(encoder, decoder).to(device)
        model_file = 'deterministic_visual_epochs_1_lr_0.001/1588434107.pt' #model_files[0]
        # load model
        model.load_state_dict(torch.load(model_dir + model_file, map_location=torch.device(device)))
    else:
        encoder = modules.Encoder(input_dim, conv_layers, z_dim)
        decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
        model_file = '/home/tom/world_models/models/variational_visual_epochs_2/lr_0.0036481/run_0/model.pt' #model_files[0]
        model = torch.load(model_file)
    
    # load data
    data = np.load(data_dir + data_files[0], allow_pickle=True)[0,1,:]
    print(data.shape) # (1000,)
    # convert 
    data = data.flatten()
    data = np.array([item for item in data])
    # reshape and normalize
    data = torch.from_numpy(np.reshape(data, (1000,3,96,96))).to(device).float() / 255

    output, _ = model.forward(data)
    output = output.detach().cpu().numpy()
    output = np.reshape(output, (1000,96,96,3))

    data = data.cpu().numpy() 
    data = data.reshape((1000,96,96,3)) 
    # Set up formatting for the movie files
    writer = ani.FFMpegWriter(fps=30)
    '''
    def init():
        im.set_data(np.zeros((96,96,3)))
        return [im]
    '''
    
    # create animations
    ''' 
    def orig_func(frame):
        im.set_array(data[frame])
        return [im]
    fig1=plt.figure()
    original_ani = ani.FuncAnimation(fig=fig1, func=orig_func, init_func=init, frames=1000)
    original_ani.save('/home/tom/world_models/plots/input.mp4')
    plt.show()
    plt.close()

    def output_func(frame):
        im.set_array(output[frame])
        return [im]
    fig2=plt.figure()
    autoencoded_ani = ani.FuncAnimation(fig=fig2, func=output_func, init_func=init, frames=1000)
    autoencoded_ani.save('/home/tom/world_models/plots/output.mp4')
    plt.show()
    plt.close()
    '''
    fig = plt.figure()
    im = plt.imshow(np.zeros((96,96,3)), animated=True)
    with writer.saving(fig, '/home/tom/world_models/plots/input.mp4', 640):
        for i in range(1000):
            im.set_data(data[i])
            writer.grab_frame()
            print(i)
    plt.close()
    fig = plt.figure()
    im = plt.imshow(np.zeros((96,96,3)), animated=True)
    with writer.saving(fig, '/home/tom/world_models/plots/output.mp4', 640):
        for i in range(1000):
            im.set_data(output[i])
            writer.grab_frame()
            print(i)
    plt.close()
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[32, 4], [64,4], [128,4], [256,4]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--deconv_layers', type=int, default=[[128, 4], [64,4], [32,4], [8,4], [3,6]], help='List of Deconv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    

    config = parser.parse_args()

    # Train the model
    main(config)
