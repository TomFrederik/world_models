"""
In this script we train the visual world model.
"""

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# other ptyhon modules
import argparse
from time import time
from datetime import datetime
import numpy as np
import os

# my modules
import models


def get_batches(data, batch_size):
    ''' 
    params:
    data: np array of shape (N_rollouts, N_steps) each element is of shape (96,96,3)
    batch_size: int
    
    returns:
    batches: N/batch_size batches of shape (batch_size, dim_of_image)
    '''

    data = data.flatten()
    data = np.array([item for item in data])
    
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


def train(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    z_dim = config.latent_dim
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    id_str = 'visual_epochs_{}_lr_{}_time{}'.format(epochs, learning_rate, time())
    
    writer = SummaryWriter(cur_dir + config.model_dir + id_str)

    # set up model
    encoder = models.Encoder(input_dim, conv_layers, z_dim)
    # not sure if I somehow have to change dimensions of the conv layers here
    decoder = models.Decoder(input_dim, conv_layers, z_dim)
    model = models.VAE(encoder, decoder).to(device)

    # init crit and optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # get data
    print('Loading data..')
    data_dir = cur_dir + '/data/'
    (_,_,files) = os.walk(data_dir).__next__()

    files = [data_dir + file for file in files]

    print('Starting training...')
    log_ctr = 0
    running_loss = 0
    file_run_ctr = 0

    for file_idx, file in enumerate(files):
        
        data = np.load(file, allow_pickle = True)
        print('Getting batches of file {}...'.format(file_idx+1))
        batches = np.array(get_batches(data[:,1,:], batch_size)) # only look at observations
        # shape is e.g. (781, 128, 3, 96, 96) = (nbr_batches, batch_size, C_in, H, W)

        for epoch in range(epochs):
            
            for step, batch_input in enumerate(batches):
                
                # store batches on GPU
                # make tensor
                batch_input = torch.from_numpy(batch_input).to(device)
            
                # make float
                batch_input = batch_input.float()
            
                # set grad to zero
                optimizer.zero_grad()

                # forward pass
                batch_output = model.forward(batch_input)

                # compute loss
                loss = criterion(batch_output, batch_input)

                # backward pass
                loss.backward()

                # updating weights
                optimizer.step()
                
                # update lr
                scheduler.step()

                ###
                # logging
                ###
                running_loss += loss.item()

                # inc log counter
                log_ctr += 1
            
                if log_ctr % 10 == 0:
                    # log the losses
                    writer.add_scalar('training loss',
                                running_loss / 10,
                                epoch * batches.shape[0] + file_run_ctr + step)
                    print('At epoch {0:5d}, step {1:5d}, the loss is {2:4.10f}'.format(epoch+1, epoch * batches.shape[0] + file_run_ctr + step+1, running_loss/10))
                    running_loss = 0
        
        # inc step counter across files
        file_run_ctr += batches.shape[0]*epochs
        
        #free memory for next file
        del data
        del batches

    print('Done training.')

    print('Saving Model..')
    torch.save(model.state_dict(), cur_dir + config.model_dir + id_str + '.pt')
    print("Model saved. Exiting program..")
################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[10, 3], [10,3], [10,3]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    

    config = parser.parse_args()

    # Train the model
    train(config)
