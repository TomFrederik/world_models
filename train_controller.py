"""
In this script we train the controller model
"""

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

# parallel processing
import ray
#import torch.multiprocessing as mp

# gym modules
import gym
from gym import wrappers

# stable baselines
#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
#from stable_baselines import PPO2

# other ptyhon modules
import argparse
from time import time
from datetime import datetime
import numpy as np
import os

# my modules
import modules
from CMA_ES import CMA_ES


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    learning_rate = config.learning_rate
    epochs = config.epochs
    pop_size = config.pop_size
    num_parallel_agents = config.num_parallel_agents
    selection_pressure = config.selection_pressure
    ctrl_layers = config.ctrl_layers
    stop_crit = config.stop_crit

    if torch.cuda.is_available():
        device = 'cuda:0'
        ctrl_device = 'cuda:0'
    else:
        device = 'cpu'
        ctrl_device = 'cpu'
    
    ## DEBUGGING
    #device = 'cpu'
    #ctrl_device = 'cpu'
    ###
    # print(mp.cpu_count()) # 2


    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = cur_dir + config.model_dir

    id_str = 'ctrl_epochs_{}_lr_{}_popsize_{}'.format(epochs, learning_rate, pop_size)
    
    #writer = SummaryWriter(model_dir + id_str)

    print('Setting up world model..')

    # set up visual model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    vis_model = modules.VAE(encoder, decoder, encode_only=True).to(device)
    
    # load visual model
    vis_model_file = 'variational_visual_epochs_1_lr_0.001/1588429800.pt'
    vis_model.load_state_dict(torch.load(model_dir + vis_model_file, map_location=torch.device(device)))
    vis_model.eval()
    
    # load mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params).to(device)
    mdn_model_file = 'mdnrnn_epochs_20_lr_0.001_layers_100_100_50_50_schedsteps_100.pt'
    mdn_model.load_state_dict(torch.load(model_dir + mdn_model_file, map_location=torch.device(device)))
    mdn_model.eval()

    print('Setting up CMA-ES..')
    # set up CMA-ES
    # parameters for control network
    ctrl_kwargs = {
        'input_dim':lstm_units+z_dim,
        'layers':ctrl_layers,
        'ac_dim':3
    }

    # parameters for CMA
    CMA_parameters = {
        'model_class':modules.Controller,
        'ctrl_device':ctrl_device,
        'vis_model':vis_model,
        'mdn_rnn':mdn_model,
        'model_kwargs':ctrl_kwargs, 
        'env_id':'CarRacing-v0', 
        'num_parallel_agents':num_parallel_agents, 
        'pop_size':pop_size,
        'selection_pressure':selection_pressure,
        'model_dir':model_dir+'/'+id_str+'/'
    }

    CMA = CMA_ES(**CMA_parameters)
    
    print('Starting training...')

    # init parallel processing
    if num_parallel_agents > 1:
        ray.init(num_cpus=num_parallel_agents, 
                object_store_memory=1024*1024*1024*13,
                redis_max_memory=1024*1024*100
        )

    # train
    best_parameters = CMA.train(stop_crit=stop_crit)
    print(best_parameters.shape)
    print(best_parameters)
    # save model
    best_model = modules.Controller(ctrl_kwargs)
    best_model.parameters().data = best_parameters
    torch.save(best_model, model_dir + 'controller_{}.pt'.format(int(time())))

    #print('Parameters in the VAE: ', count_parameters(vis_model))
    #print('Parameters in the MDN_RNN: ', count_parameters(mdn_model))
    #print('Parameters in the Controller: ', count_parameters(controller))



    
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
    parser.add_argument('--mdn_layers', type=int, default=[100,100,50,50], help='List of layers in the MDN')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for mixture model')
    parser.add_argument('--ctrl_layers', type=int, default=[], help='List of layers in the Control network')
    parser.add_argument('--pop_size', type=int, default=1000, help='Population size for CMA-ES')
    parser.add_argument('--num_parallel_agents', type=int, default=8, help='Number of agents run in parallel when evaluating fitness')
    parser.add_argument('--selection_pressure', type=float, default=0.9, help='Percentage of population that survives each iteration')
    parser.add_argument('--stop_crit', type=int, default=600, help='Average fitness value that needs to be reached')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
