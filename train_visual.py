"""
In this script we train the visual world model.
"""

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# other ptyhon modules
import argparse
import time
from datetime import datetime
import numpy as np

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
        # zero pad data, leads to memory error
        '''
        pad_shape = batch_size - (data.shape[0] % batch_size)
        print((pad_shape,) + data.shape[1:]) # (pad_shape, 96, 96, 3)
        data = np.append(data, np.zeros((pad_shape,) + data.shape[1:]))
        '''
        # drop last x frames
        drop_shape = data.shape[0] % batch_size
        data = data[:-drop_shape]

    if data.shape[0] % batch_size != 0:
        raise ValueError("Padding didn't work")

    batches = np.split(data, data.shape[0]/batch_size)

    return np.array(batches)


def train(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    z_dim = config.latent_dim
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    train_steps = config.train_steps
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    # set up model
    encoder = models.Encoder(input_dim, conv_layers, z_dim)
    # not sure if I somehow have to change dimensions of the conv layers here
    decoder = models.Decoder(input_dim, conv_layers, z_dim)
    model = models.VAE(encoder, decoder).cuda()

    # init crit and optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    print('Loading data..')
    # get data
    data = np.load('/home/tom/data/random_rollouts_0_500.npy', allow_pickle=True) # 500 2-tuples (action, observation)
    print('Getting batches...')
    batches = get_batches(data[:,1,:], batch_size)

    # loss array
    losses = []
    
    print('Starting training...')
    for step, batch_input in enumerate(batches):

        # store batches on GPU
        # make tensor
        batch_input = torch.from_numpy(batch_input).cuda()
      
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

        # log the losses
        losses.append(loss.item())

        if step % 100 == 0:
            print("Train step {}/{} , Loss = {}".format(step, train_steps, loss.item()))

    print('Done training.')

    print('Saving Model..')
    torch.save(model.state_dict(), "visual_model_v0")
    print("Model saved. Exiting program..")
################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[100, 3], [100,3]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")

    config = parser.parse_args()

    # Train the model
    train(config)
