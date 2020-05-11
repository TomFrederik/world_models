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
import gc
import functools as ft

# my modules
import modules


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


def train_visual(config):

    # get arguments
    input_dim = config['input_dim']
    conv_layers = config['conv_layers']
    deconv_layers = config['deconv_layers']
    z_dim = config['latent_dim']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # use AE or VAE?
    deterministic = False
    
    if deterministic:
        id_str = 'deterministic_visual_epochs_{}'.format(epochs)
    else:
        id_str = 'variational_visual_epochs_{}'.format(epochs)
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = cur_dir+config['model_dir'] +id_str+'/lr_{}/'.format(learning_rate)
    (_,_,files) = os.walk(log_dir).__next__()

    run_id = 'run_' + str(len(files))
    log_dir = log_dir + run_id
    writer = SummaryWriter(log_dir)
    

    # set up model
    if deterministic:
        encoder = modules.Det_Encoder(input_dim, conv_layers, z_dim)
        decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
        model = modules.AE(encoder, decoder).to(device)
    else:
        encoder = modules.Encoder(input_dim, conv_layers, z_dim)
        decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
        model = modules.VAE(encoder, decoder).to(device)

    # (re-)init crit and optim
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    criterion = nn.MSELoss()
    
    # get data
    print('Loading data..')
    data_dir = cur_dir + '/data/'
    (_,_,files) = os.walk(data_dir).__next__()
    files = [data_dir + file for file in files]
    
    # train model on all files but the last one
    model.train()
    model = train(model, deterministic, optimizer, scheduler, criterion, writer, files, batch_size, device, epochs, log_dir)
    print('Done training model.')
    print('Saving model')
    torch.save(model, log_dir+'/model.pt')

def train(model, deterministic, optimizer, scheduler, criterion, writer, files, batch_size, device, epochs, log_dir):
    print('Starting training...')
    log_ctr = 0
    running_loss = 0
    for epoch in range(epochs):
        for file_idx, file in enumerate(files):
            
            data = np.load(file, allow_pickle = True)
            print('Getting batches of file {}...'.format(file_idx+1))
            batches = np.array(get_batches(data[:,1,:], batch_size)) # only look at observations
            # shape is e.g. (781, 128, 3, 96, 96) = (nbr_batches, batch_size, C_in, H, W)

        
            
            for step, batch_input in enumerate(batches):
                
                # store batches on GPU
                # make tensor
                batch_input = torch.from_numpy(batch_input).to(device)
            
                # make float
                batch_input = batch_input.float()

                # normalize from 0..255 to 0..1
                batch_input = batch_input / 255
            
                # set grad to zero
                optimizer.zero_grad()

                # forward pass
                if deterministic:
                    batch_output = model.forward(batch_input)
                    loss = criterion(batch_output, batch_input)
                else:
                    batch_output, kl_loss = model.forward(batch_input)
                    loss = criterion(batch_output, batch_input)
                    loss += kl_loss
                    
                # backward pass
                loss.backward()

                
                del batch_input
                del batch_output
                gc.collect()
                

                # updating weights
                optimizer.step()
                
                # update lr
                scheduler.step()

                ###
                # logging
                ###
                

                # inc log counter
                log_ctr += 1
                
                running_loss += loss.item()
                if log_ctr % 10 == 0:
                    # log the losses
                    writer.add_scalar('training loss',
                               running_loss / 10,
                               log_ctr)
                    writer.flush()
                    print('At epoch {0:3d}, file {1:3d}, step {2:5d}, the loss is {3:4.10f}'.format(epoch+1, file_idx+1, step+1, running_loss/10))
                    running_loss = 0
        
            #free memory for next file
            del data
            del batches
            gc.collect()

            # save model
            print('Saving model...')
            torch.save(model, log_dir+'/model.pt')

    return model
    '''
    # save progress so far!
    print('Saving Model..')
    torch.save(model.state_dict(), cur_dir + config['model_dir'] + id_str + '/{}.pt'.format(start_time))
    print("Model saved.")
    '''
    
################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[32, 4], [64,4], [128,4], [256,4]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--deconv_layers', type=int, default=[[128, 4], [64,4], [32,4], [8,4], [3,6]], help='List of Deconv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=3.6481e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = vars(parser.parse_args())

    train_visual(config)
