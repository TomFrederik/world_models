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

    if mean.is_cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    #print('var')
    #print(var)
    
    # inverse covariance matrix, is diagonal so is simple
    inv_var = 1/var

    #print('inv var')
    #print(inv_var, inv_var.shape)
    
    # compute gaussians
    mean = mean.reshape((mean.shape[2], mean.shape[0], mean.shape[1], mean.shape[3])) # (5, batch_size, seq_len, 32)
    #print('mean')
    #print(mean.shape)
    
    diff_sq = torch.add(mean, -target)**2
    diff_sq = diff_sq.reshape((diff_sq.shape[1], diff_sq.shape[2], diff_sq.shape[0], diff_sq.shape[3]))
    diff_sq = torch.mul(diff_sq, inv_var)
    diff_sq = torch.sum(diff_sq, dim=-1)
    #print('diff_sq:',diff_sq, diff_sq.shape)
    
    exponent = -0.5 * diff_sq
    #print('exponent')
    #print(exponent)
    #print(exponent.shape)
    
    det_inv_var = torch.prod(inv_var, dim=-1)
    
    gaussian = 1/((2*PI)**16) * det_inv_var * torch.exp(exponent)
    #print('gaussian')
    #print(gaussian)
    #print('exponential')
    #print(torch.exp(exponent))
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

    if batch_size > 100:
        raise ValueError('batch_size should be equal or less than 100, but is {}'.format(batch_size))

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


    id_str = 'mdnrnn_epochs_{}/lr_{}/temp_{}'.format(epochs, learning_rate, config.temp)
    
    log_dir = model_dir + id_str
    (_,dirs,_) = os.walk(log_dir).__next__()
    run_id = 'run_' + str(len(dirs))
    log_dir = log_dir + '/' + run_id

    writer = SummaryWriter(log_dir)

    # set up mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params).to(device)

    # init optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mdn_model.parameters()), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_steps, gamma=0.95)

    # get data
    print('Loading data..')
    ac_dir = '/home/tom/disk_1/world_models_data/random_rollouts/'
    enc_dir = '/home/tom/disk_1/world_models_data/enc_data/'
    (_,_,ac_files) = os.walk(ac_dir).__next__()

    enc_files = [enc_dir + 'encoded_images_' + str(i) + '.npy' for i in range(len(ac_files))]
    ac_files = sorted([ac_dir + file for file in ac_files])

    print('Starting training...')
    log_ctr = 0
    global_ctr = 0
    running_loss = 0
    file_idx = 0

    for obs_file, ac_file in zip(enc_files,ac_files):

        # fetch data
        print('Getting batches of file {}...'.format(file_idx+1))

        obs_data = np.load(obs_file, allow_pickle=True)
        ac_data = np.load(ac_file, allow_pickle = True)[:,0,:]
        obs_batches = np.array(get_obs_batches(obs_data, batch_size)) # only look at observations
        act_batches = np.array(get_act_batches(ac_data, batch_size)) # only look at actions
        batches_in = np.append(obs_batches[:,:,:-1,:], act_batches[:,:,1:,:], axis=-1) # is of shape (nbr_batches, batch_size, nbr_frames, z_dim+ac_dim)
        batches_target = torch.from_numpy(obs_batches[:,:,1:,:]).float()

        # free up memory
        del obs_batches
        del act_batches
        del obs_data
        del ac_data

        for epoch in range(epochs):
            
            for step in range(batches_in.shape[0]):
                # make batch tensor and float
                batch_input = torch.from_numpy(batches_in[step]).float().to(device)
                
                # set grad to zero
                optimizer.zero_grad()

                # forward pass
                coeff, mean, var = mdn_model.forward(batch_input)

                # compute loss
                loss = mdn_loss(input=batch_input, target=batches_target[step].to(device), coeff=coeff, mean=mean, var=var)
                #print(loss.item())
                
                # backward pass
                #time1 = time()
                loss.backward()
                #print('time: ', time()-time1)
                
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
                global_ctr += 1
            
                if log_ctr % 10 == 0:
                    # log the losses
                    writer.add_scalar('training loss',
                                running_loss / 10,
                                global_ctr)
                    writer.flush()
                    print('At epoch {0:5d}, step {1:5d}, global_step {2:5d}, the loss is {3:4.10f}'.format(epoch+1, step+1, global_ctr, running_loss/10))
                    running_loss = 0
        
        #free memory for next file
        del batches_in

        # save progress so far
        print('Saving Model..')
        torch.save(mdn_model.state_dict(), log_dir + '/model.pt')
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
    parser.add_argument('--temp', type=float, default=2, help='Temperature for mixture model')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
