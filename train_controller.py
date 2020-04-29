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

# gym modules
import gym
from gym import wrappers

# other ptyhon modules
import argparse
from time import time
from datetime import datetime
import numpy as np
import os

# my modules
import modules



def train(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    deconv_layers = config.deconv_layers
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

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = cur_dir + config.model_dir

    id_str = 'ctrl_epochs_{}_lr_{}_time_{}'.format(epochs, learning_rate, time())
    
    writer = SummaryWriter(model_dir + id_str)

    # set up visual model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    # not sure if I somehow have to change dimensions of the conv layers here
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    vis_model = modules.VAE(encoder, decoder).to(device)
    
    # load visual model
    vis_model_file = 'visual_epochs_1_lr_0.001_time1587912855.6904068.pt'
    vis_model.load_state_dict(torch.load(model_dir + vis_model_file, map_location=torch.device(device)))

    # load mdn model
    mdn_params = {'input_dim':z_dim, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params)
    mdn_model_file = ''
    mdn_model.load_state_dict(torch.load(model_dir + mdn_model_file, map_location=torch.device(device)))

    # init crit and optim
    criterion = $$$
    optimizer = optim.Adam(mdn_model.parameters(), lr = learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    print('Starting training...')
    log_ctr = 0
    running_loss = 0
    file_run_ctr = 0

    
################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[32, 4], [64,4], [128,4], [256,4]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--deconv_layers', type=int, default=[[128, 4], [64,4], [32,4], [8,4], [3,6]], help='List of Deconv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of layers in the LSTM')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units per layer')
    parser.add_argument('--nbr_gauss', type=int, default=5, help='Number of gaussians for MDN')
    parser.add_argument('--mdn_layers', type=int, default=[100,100], help='List of layers in the MDN')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for mixture model')
    parser.add_argument('--ctrl_layers', type=int, default=[], help='List of layers in the Control network')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
