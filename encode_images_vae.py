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

# my modules
import modules

def get_obs_batches(data, batch_size):
    ''' 
    params:
    data: np array of shape (N_rollouts, N_steps) each element is of shape (96,96,3)
    batch_size: int
    
    returns:
    batches: N/batch_size batches of shape (batch_size, dim_of_image)
    '''

    data = data.flatten()
    
    # get first item
    new_data = data[0]
    new_data = np.reshape(new_data, (1, *new_data.shape))
    data = data[1:]

    # convert to lists
    data = list(data)
    new_data = list(new_data)

    # loop through data and add
    for i in range(len(data)):
        new_data.append(data.pop())
        

    data = new_data
    del new_data
    data = np.array(data)

    # reshape from (N, 96, 96 , 3) to (N, 3, 96, 96)
    data = np.reshape(data, (data.shape[0], data.shape[-1], data.shape[1], data.shape[2]))

    if data.shape[0] % batch_size != 0:
        # drop last x frames
        drop_n = data.shape[0] % batch_size
        data = data[:-drop_n]

    if data.shape[0] % batch_size != 0:
        raise ValueError("Dropping didn't work")

    batches = np.split(data, data.shape[0]/batch_size)

    return batches

def main(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    deconv_layers = config.deconv_layers
    z_dim = config.latent_dim
    batch_size = config.batch_size

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    cur_dir = os.path.dirname(os.path.realpath(__file__))

    model_dir = cur_dir + '/models/'
    data_dir = cur_dir + '/data/'
    enc_data_dir = cur_dir + '/enc_data/'

    (_,_,model_files) = os.walk(model_dir).__next__()
    (_,_,data_files) = os.walk(data_dir).__next__()

    # set up model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    # not sure if I somehow have to change dimensions of the conv layers here
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    model = modules.VAE(encoder, decoder, encode_only=True).to(device)
    
    # load model
    model_file = 'visual_epochs_1_lr_0.001_time1587912855.6904068.pt' #model_files[0]
    model.load_state_dict(torch.load(model_dir + model_file, map_location=torch.device(device)))

    # set to eval mode
    model.eval()

    # get data
    print('Loading data..')
    data_dir = cur_dir + '/data/'
    (_,_,files) = os.walk(data_dir).__next__()

    files = sorted([data_dir + file for file in data_files])

    print('Starting run...')

    for file_idx, file in enumerate(files):
        
        print('Getting batches of file {}...'.format(file_idx+1))
        file_out = np.zeros((100, 1000, z_dim))
        data = np.load(file, allow_pickle = True)[:,1,:]
        # shape of batches is e.g. (781, 128, 3, 96, 96) = (nbr_batches, batch_size, C_in, H, W)
        batches = np.array(get_obs_batches(data, batch_size)) # only look at observations

        for step in range(len(batches)):
            
            # pop batches
            batch_input, batches = batches[0], batches[1:]

            # store batches on GPU
            # make tensor
            batch_input = torch.from_numpy(batch_input).to(device)
        
            # make float
            batch_input = batch_input.float()

            # normalize from 0..255 to 0..1
            batch_input = batch_input / 255

            # forward pass
            batch_outpu, _ = model.forward(batch_input)

            run_id = (step*batch_size) // 1000
            frame_in_run = (step*batch_size) % 1000

            file_out[run_id, frame_in_run:frame_in_run+batch_size] = batch_output.cpu().detach().numpy()
        
        del data
        del batches


        print('Saving data of file {}'.format(file_idx+1))
        
        np.save(enc_data_dir+'encoded_images_{}.npy'.format(file_idx), file_out)
        del file_out
    
    

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[32, 4], [64,4], [128,4], [256,4]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--deconv_layers', type=int, default=[[128, 4], [64,4], [32,4], [8,4], [3,6]], help='List of Deconv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size")
    

    config = parser.parse_args()

    # Train the model
    main(config)
