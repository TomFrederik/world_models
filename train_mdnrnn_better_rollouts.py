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
import modules

# constants
PI = 3.14159265359

def mdn_loss(input, target, coeff, mean, var):

    bs = input.shape[0] #batch size
    #print('var 3')
    #print(var[0,0])
    # inverse covariance matrix, is diagonal so is simple
    inv_var = 1/var

    # copy target to process for all kernels at the same time
    dummy = torch.zeros_like(mean)
    for i in range(dummy.shape[2]):
        dummy[:,:,i,:] = target
    target = dummy
    del dummy

    # compute gaussians
    diff_sq = (target - mean)**2
    #print('diff_sq:',diff_sq, diff_sq.shape)
    exponent = -0.5 * inv_var**64 * torch.sum(diff_sq, dim=-1) 
    #print('exponent')
    #print(exponent)
    #print(exponent.shape)
    
    #print('inv var')
    #print(inv_var, inv_var.shape)
    gaussian = 1/((2*PI)**16) * inv_var**32 * torch.exp(exponent)
    #print('gaussian')
    #print(gaussian)
    #print(gaussian.shape)
    # multiply by coefficients and sum over all gaussians
    likelihood = torch.sum(coeff * gaussian, dim=-1) # shape is now (batch_size, seq_len)
    log_likelihood = -1*torch.log(likelihood)
    #print('log_likelihood')
    #print(log_likelihood)
    #print(log_likelihood.shape)
    
    # now average over all data points
    loss = torch.mean(log_likelihood)

    return loss
    

def get_obs_batches(data, batch_size):
    ''' 
    params:
    data: np array of shape (N_rollouts, N_steps, 32)
    batch_size: int
    
    returns:
    batches: N/batch_size batches of shape (batch_size, 32)
    '''

    # drop last x runs to make it divisable by batch_size
    if data.shape[0] % batch_size != 0:
        # drop last x frames
        drop_n = data.shape[0] % batch_size
        data = data[:-drop_n]

    if data.shape[0] % batch_size != 0:
        raise ValueError("Dropping didn't work")

    batches = np.split(data, data.shape[0]/batch_size)

    return batches

def get_act_batches(data, batch_size):
    ''' 
    params:
    data: np array of shape (N_rollouts, N_steps) each element is a list of shape (3)
    batch_size: int
    
    returns:
    batches: N/batch_size batches of shape (batch_size, 3)
    '''

    # reshape (N, 1000) array with lists of 3 to (N, 1000, 3) array
    new_data = np.zeros((*data.shape, 3))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_data[i,j] = data[i][j]
    
    data = new_data
    del new_data

    # drop last x runs to make it divisable by batch_size
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
    z_dim = config.latent_dim
    lstm_units = config.lstm_units
    lstm_layers = config.lstm_layers
    mdn_layers = config.mdn_layers
    nbr_gauss = config.nbr_gauss
    temp = config.temp
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    sched_steps = 100

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = cur_dir + config.model_dir

    layer_str = ''
    for i in range(len(mdn_layers)):
        layer_str += str(mdn_layers[i])+'_'


    id_str = 'better_mdnrnn_epochs_{}_lr_{}_layers_{}temp_{}_schedsteps_{}'.format(epochs, learning_rate, layer_str, config.temp, sched_steps, time())
    
    writer = SummaryWriter(model_dir + id_str)

    # set up mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params).to(device)

    # init optimizer and scheduler
    optimizer = optim.Adam(mdn_model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_steps, gamma=0.95)

    # get data
    print('Loading data..')
    ac_dir = cur_dir + '/data/'
    enc_dir = cur_dir + '/enc_data/'
    (_,_,ac_files) = os.walk(ac_dir).__next__()

    enc_files = [enc_dir + 'encoded_images_' + str(i) + '.npy' for i in range(len(ac_files))]
    ac_files = sorted([ac_dir + file for file in ac_files])

    print('Starting training...')
    log_ctr = 0
    running_loss = 0
    file_run_ctr = 0
    file_idx = 0

    for obs_file, ac_file in zip(enc_files,ac_files):

        # fetch data
        print('Getting batches of file {}...'.format(file_idx+1))

        obs_data = np.load(obs_file, allow_pickle=True)
        ac_data = np.load(ac_file, allow_pickle = True)[:,0,:]
        obs_batches = np.array(get_obs_batches(obs_data, batch_size)) # only look at observations
        act_batches = np.array(get_act_batches(ac_data, batch_size)) # only look at actions
        batches = np.append(obs_batches, act_batches, axis=-1) # is of shape (nbr_batches, batch_size, nbr_frames, z_dim+ac_dim)
        
        # free up memory
        del obs_batches
        del act_batches
        del obs_data
        del ac_data

        for epoch in range(epochs):
            
            for step, batch_input in enumerate(batches):
                
                # make batch tensor and float
                batch_input = torch.from_numpy(batch_input).float().to(device)
            
                # set grad to zero
                optimizer.zero_grad()

                # forward pass
                coeff, mean, var = mdn_model.forward(batch_input)

                # compute loss
                loss = mdn_loss(input=batch_input, target=batch_input[:,1:,:32], coeff=coeff, mean=mean, var=var)
                print(loss.item())
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
        del batches

        # save progress so far
        print('Saving Model..')
        torch.save(mdn_model.state_dict(), cur_dir + config.model_dir + id_str + '.pt')
        print("Model saved.")
        file_idx += 1

    print('Done training.')
    print('Exiting program..')
################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of layers in the LSTM')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units per layer')
    parser.add_argument('--nbr_gauss', type=int, default=5, help='Number of gaussians for MDN')
    parser.add_argument('--mdn_layers', type=int, default=[100,100,50,50], help='List of layers in the MDN')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for mixture model')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
