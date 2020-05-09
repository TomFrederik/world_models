"""
In this script we train the controller model
"""

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

def train_CMA(CMA, writer, stop_crit=600, old_run={'mean':None,'cov':None,'run_nbr':None}):
        '''
        Executes the CMA-ES algorithm.
        params:
        CMA - an instance of CMA
        stop_crit - average fitness of the population that triggers stopping training
        writer - a tensorboard summary writer

        returns:
        best_candidate - the best parameter vector of the last generation with shape (num_params)
        '''

        ctr = 0
        
        mean = old_run['mean']
        cov = old_run['cov']
        run_nbr = old_run['run_nbr']

        # continuing an old run?
        if mean:
            CMA.mean = mean
            CMA.covariance = cov
            CMA.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance=cov)
            CMA.model_dir = CMA.model_dir[:-1] + str(run_nbr)

        # sample initial population
        cur_pop = CMA.sample(CMA.pop_size)

        # calc fitness of current pop
        cur_pop_fitness = CMA.fitness(cur_pop)

        mean_fitness = torch.mean(cur_pop_fitness)
        best_fitness_id = torch.argsort(cur_pop_fitness, descending=True)[0]

        writer.add_scalar(tag='mean fitness', scalar_value=mean_fitness, global_step=ctr)
        writer.add_scalar(tag='best fitness', scalar_value=cur_pop_fitness[best_fitness_id], global_step=ctr)
        writer.flush()

        print('Just completed step {0:5d}, average fitness of last step was {1:4.3f}'.format(ctr, mean_fitness))
        print('Saving best candidate..')
        best_candidate = cur_pop[best_fitness_id]
        torch.save(best_candidate, f=CMA.model_dir+'/best_candidate.pt')
        print('Saving covariance and mean..')
        torch.save(CMA.mean, f=CMA.model_dir+'/mean.pt')
        torch.save(CMA.covariance, f=CMA.model_dir+'/cov.pt')

        done = False
        while not done:
            ctr += 1
            # compute new dist and sample for new pop
            cur_pop, CMA.dist = CMA.evolution_step(cur_pop, cur_pop_fitness)

            # calc fitness of current pop
            cur_pop_fitness = CMA.fitness(cur_pop)

            # check if done
            if torch.mean(cur_pop_fitness) >= stop_crit:
                done = True

            ctr += 1

            mean_fitness = torch.mean(cur_pop_fitness)
            best_fitness_id = torch.argsort(cur_pop_fitness, descending=True)[0]

            writer.add_scalar('mean fitness', mean_fitness, ctr)
            writer.add_scalar('best fitness', cur_pop_fitness[best_fitness_id], ctr)
            writer.flush()

            print('Just completed step {0:5d}, average fitness of last step was {1:4.3f}'.format(ctr, mean_fitness))
            print('Saving best candidate..')
            best_candidate = cur_pop[best_fitness_id]
            torch.save(best_candidate, f=CMA.model_dir+'/best_candidate.pt')
            print('Saving covariance and mean..')
            torch.save(CMA.mean, f=CMA.model_dir+'/mean.pt')
            torch.save(CMA.covariance, f=CMA.model_dir+'/cov.pt')

        
        print('Completed training after {0:5d} steps. Best fitness of last step was {1:4.3f}'.format(ctr, best_candidate))
        print('Saving best candidate..')
        torch.save(best_candidate, f=CMA.model_dir+'/best_candidate.pt')
        print('Saving covariance and mean..')
        torch.save(CMA.mean, f=CMA.model_dir+'/mean.pt')
        torch.save(CMA.covariance, f=CMA.model_dir+'/cov.pt')
        return best_candidate

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

    # setting up paths
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = cur_dir + config.model_dir
    id_str = 'ctrl_epochs_{}_lr_{}_popsize_{}'.format(epochs, learning_rate, pop_size)

    # init tensorboard
    log_dir = model_dir+'/'+id_str+'/'
    (_,_,files) = os.walk(log_dir).__next__()
    run_id = 'run_' + str(len(files))
    log_dir = log_dir + run_id
    writer = SummaryWriter(log_dir)
    print('Saving results and logs in directory', log_dir)

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
        'model_dir':log_dir
    }

    CMA = CMA_ES(**CMA_parameters)
    
    print('Starting training...')
    # init parallel processing
    if num_parallel_agents > 1:
        ray.init(num_cpus=num_parallel_agents, 
                object_store_memory=1024*1024*1024*12,
                redis_max_memory=1024*1024*200
        )

    # train
    mean = None
    cov = None
    run_nbr = None
    old_run = {'mean':mean,'cov':cov,'run_nbr':run_nbr}
    best_parameters = train_CMA(CMA=CMA, writer=writer, stop_crit=stop_crit, old_run=old_run)
    
    print('Saving final model')
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
    parser.add_argument('--pop_size', type=int, default=1500, help='Population size for CMA-ES')
    parser.add_argument('--num_parallel_agents', type=int, default=14, help='Number of agents run in parallel when evaluating fitness')
    parser.add_argument('--selection_pressure', type=float, default=0.7, help='Percentage of population that survives each iteration')
    parser.add_argument('--stop_crit', type=int, default=600, help='Average fitness value that needs to be reached')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
