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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CMA_ES:
    
    def __init__(self, model_class, model_kwargs={}, env_id='CarRacing-v0', num_runs=5, pop_size=10, selection_pressure=0.1):
        
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.env_id = env_id
        self.num_runs = num_runs
        self.pop_size = pop_size
        self.selection_pressure = selection_pressure
    
    def

        
    def fitness(self, pop, model_class, model_kwargs, env_id, num_runs):
        '''
        params:
        env_id - ID for a gym environemnt
        pop - a population of parameter vectors with shape (pop_size, num_params)
        model_class - the constructor for an actor model
        model_kwargs - keyword arguments for the model
        num_runs - number of rollouts for each parameter vector to average reward

        returns:
        fitness - fitness values of each parameter vector. Is of shape (pop_size)
        '''
        pop_size = pop.shape[0]

        # create environment
        env = gym.make(env_id)

        # container for fitness values
        fitness = torch.zeros(pop_size)

        for agent_id in range(pop.shape[0]):
            agent_params = pop[agent_id,:]
            model = model_class(model_kwargs)
            print(model.parameters().data)
            
            # set model params to this agent's params
            model.parameters().data = agent_params
            
            for i in range(num_runs):
                # collect rollouts
                obs = env.reset
                action = model(obs)
                done = False
                cum_rew = 0

                while not done:
                    obs, rew, done, _ = env(action)
                    action = model(obs)
                    cum_rew += rew
                
                fitness[agent_id] += cum_rew / num_runs


        return fitness

    def sample(self, mean, covariance, pop_size):
        '''
        params:
        mean - mean vector of shape (n)
        covariance - covariance matrix of shape (n, n)
        pop_size - number of parameter vectors that should be sampled

        returns:
        sample - parameter matrix of shape (pop_size, n)
        '''

        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)
        sample = dist.rsample(sample_shape=torch.Size(pop_size, mean.shape[0]))

    def grim_reaper(self, cur_pop, selection_pressure):
        '''
        params:
        cur_pop - parameter matrix of shape (pop_size, n)
        selection_pressure - percentage of population that is allowed to survive 

        returns:
        survivors - parameter matrix of shape (num_survivors, n)
        '''

        # calculate fitness of each agent
        pop_fitness = fitness(cur_pop, model_class=$$$)

        # calculate number of survivors
        num_survivors = int(cur_pop.shape[0] * selection_pressure)

        # get IDs of best agents
        survivor_ids = torch.argsort(pop_fitness, descending=True)[:num_survivors]

        # return best agents
        survivors = cur_pop[survivor_ids,:]

        return survivors




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

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # controller runs on cpu only
    ctrl_device = 'cpu'

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = cur_dir + config.model_dir

    id_str = 'ctrl_epochs_{}_lr_{}_time_{}'.format(epochs, learning_rate, time())
    
    writer = SummaryWriter(model_dir + id_str)

    ### debugging
    ###

    fitness(pop = torch.rand((2,867)), model=modules.Controller)

    ###
    ###

    # set up visual model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    vis_model = modules.VAE(encoder, decoder).to(device)
    
    # load visual model
    vis_model_file = 'visual_epochs_1_lr_0.001_time1587912855.6904068.pt'
    vis_model.load_state_dict(torch.load(model_dir + vis_model_file, map_location=torch.device(device)))

    # load mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params)
    #mdn_model_file = 'mdnrnn_epochs_20_lr_0.003_layers_4_time_1588173769.531019.pt'
    mdn_model_file = 'mdnrnn_epochs_20_lr_0.003_layers_5_schedsteps_50_time_1588180535.0663602.pt'
    mdn_model.load_state_dict(torch.load(model_dir + mdn_model_file, map_location=torch.device(device)))

    # set up controller model
    controller = modules.Controller(in_dim=256+32, layers=[], ac_dim=3).to(ctrl_device)

    #print('Parameters in the VAE: ', count_parameters(vis_model))
    #print('Parameters in the MDN_RNN: ', count_parameters(mdn_model))
    #print('Parameters in the Controller: ', count_parameters(controller))

    raise NotImplementedError
    # set up environment
    env = gym.make('CarRacing-v0')
    

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
    parser.add_argument('--mdn_layers', type=int, default=[50,50,50,50,50], help='List of layers in the MDN')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for mixture model')
    parser.add_argument('--ctrl_layers', type=int, default=[], help='List of layers in the Control network')
    parser.add_argument('--pop_size', type=int, default=20, help='Population size for CMA-ES')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
