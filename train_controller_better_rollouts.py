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
import functools

# my modules
import modules
from CMA_ES import CMA_ES


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_params(model, params):
    last_id = 0
    for layer in model.layers:
        weight_shape = layer.weight.shape
        num_weights = weight_shape[0]*weight_shape[1]
        bias_shape = layer.bias.shape
        num_biases = bias_shape[0]
        
        layer.weight.data = torch.reshape(params[last_id:last_id+num_weights], weight_shape).float()
        last_id += num_weights
        layer.bias.data = torch.reshape(params[last_id:last_id+num_biases], bias_shape).float()
        last_id += num_biases
    
    return model


def showcase(vis_model, mdn_model, agent, env_id, obs_device, ctrl_device):
    ''' showcases performance of agent'''
    while True:
        with torch.no_grad():
            env = gym.make(env_id)
            done = False
            cum_rew = 0
            step = 1
            start_time = time()    
            
            obs = env.reset()
            obs = np.reshape(obs, (1,3,96,96))
            obs = torch.from_numpy(obs).to(obs_device)
            obs = obs.float() / 255
            
            # first pass through world model
            vis_out, _ = vis_model(obs) 
            mdn_hidden = mdn_model.intial_state(batch_size=vis_out.shape[0]).to(obs_device)
            mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
            
            # first pass through controller
            ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(ctrl_device)

            action = agent(ctrl_in)
            action = torch.squeeze(action)

            while not done:
                
                obs, rew, done, _ = env.step(action.cpu().detach().numpy().astype(int))
                cum_rew += rew

                '''
                if step % 100 == 0:
                    print('Steps completed in this run: ',step)
                    duration = time()-start_time
                    print('Time since start: {} minutes and {} seconds.'.format(duration//60,duration%60))
                '''

                obs = np.reshape(obs, (1,3,96,96))
                obs = torch.from_numpy(obs).to(obs_device)
                obs = obs.float() / 255
                
                # pass through world model
                vis_out, _ = vis_model(obs)

                mdn_in = torch.unsqueeze(torch.cat([vis_out, torch.unsqueeze(action, dim=0)], dim=1), dim=1)
                mdn_hidden = mdn_model.forward(mdn_in, h_0=torch.unsqueeze(mdn_hidden, dim=0))
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(ctrl_device)

                # first pass through controller
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(ctrl_device)

                torch.cuda.empty_cache()
                action = agent(ctrl_in)
                action = torch.squeeze(action)
                
                # inc step counter
                step += 1

            env.close()

@ray.remote#(num_gpus=0.0625)
def run_agent(model, id, env_id, ctrl_device, obs_device, total_start_time, vis_model, mdn_model):
    with torch.no_grad():
        env = gym.make(env_id)
        done = False
        cum_rew = 0
        step = 1
        start_time = time()    
        
        obs = env.reset()
        obs = np.reshape(obs, (1,3,96,96))
        obs = torch.from_numpy(obs).to(obs_device)
        obs = obs.float() / 255
        
        # first pass through world model
        vis_out, _ = vis_model(obs) 
        mdn_hidden = mdn_model.intial_state(batch_size=vis_out.shape[0]).to(obs_device)
        mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
        
        # first pass through controller
        ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(ctrl_device)

        action = model(ctrl_in)
        action = torch.squeeze(action)

        while not done:
            
            obs, rew, done, _ = env.step(action.cpu().detach().numpy().astype(int))
            cum_rew += rew

            '''
            if step % 100 == 0:
                print('Steps completed in this run: ',step)
                duration = time()-start_time
                print('Time since start: {} minutes and {} seconds.'.format(duration//60,duration%60))
            '''

            obs = np.reshape(obs, (1,3,96,96))
            obs = torch.from_numpy(obs).to(obs_device)
            obs = obs.float() / 255
            
            # pass through world model
            vis_out, _ = vis_model(obs)

            torch.cuda.empty_cache()
            mdn_in = torch.unsqueeze(torch.cat([vis_out, torch.unsqueeze(action, dim=0)], dim=1), dim=1)
            mdn_hidden = mdn_model.forward(mdn_in, h_0=torch.unsqueeze(mdn_hidden, dim=0))
            mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(ctrl_device)

            # first pass through controller
            ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(ctrl_device)

            torch.cuda.empty_cache()
            action = model(ctrl_in)
            torch.cuda.empty_cache()
            action = torch.squeeze(action)
            
            # inc step counter
            step += 1

        env.close()            
        duration = time()-start_time
        total_duration = time() - total_start_time
        '''
        print('\n')
        print('###################')
        print('\n\n')
        print('Finished rollout with id {}. Duration: {} minutes and {} seconds'.format(id, duration//60, duration%60))
        print('Total time elapsed since start of this generation: {} minutes and {} seconds'.format(total_duration//60, total_duration%60))
        print('\n\n')
        print('###################')
        print('\n')
        '''
        #print('Aget with id {} had cum_rew of {}'.format(id, cum_rew))
        return 1000 - cum_rew


def fitness(pop, ctrl_kwargs, model_class, env_id, ctrl_device, obs_device, vis_model, mdn_model):
    '''
    params:
    pop - a population of parameter vectors with shape (pop_size, num_params)
    
    returns:
    fitness - fitness values of each parameter vector. Is of shape (pop_size)
    
    '''
    
    total_start_time = time()

    pop_size = pop.shape[0]

    # container for fitness values
    fitness = torch.zeros(pop_size)
    
    # list of all the models instantiated with their respective parameters
    models = [model_class(**ctrl_kwargs) for _ in range(pop_size)]
    for run_id in range(pop_size):
        #load params
        last_id = 0
        for layer in models[run_id].layers:
            weight_shape = layer.weight.shape
            num_weights = weight_shape[0]*weight_shape[1]
            bias_shape = layer.bias.shape
            num_biases = bias_shape[0]
            
            layer.weight.data = torch.reshape(pop[run_id,last_id:last_id+num_weights], weight_shape).float()
            last_id += num_weights
            layer.bias.data = torch.reshape(pop[run_id,last_id:last_id+num_biases], bias_shape).float()
            last_id += num_biases

    
    # run each agent once
    cum_rew_ids = [run_agent.remote(model, id, env_id, ctrl_device, obs_device, total_start_time, vis_model, mdn_model) for (model, id) in zip(models, np.arange(pop_size))]
    
    # save as fitness and return
    for run_id in range(pop_size):
        fitness[run_id] = ray.get(cum_rew_ids[run_id])
            
    return fitness

def train_CMA(CMA, writer, stop_crit=0.002, old_run={'mean':None,'cov':None,'run_nbr':None}, model_dir=None):
        '''
        Executes the CMA-ES algorithm.
        params:
        CMA - an instance of CMA
        stop_crit - average fitness of the population that triggers stopping training
        writer - a tensorboard summary writer

        returns:
        best_candidate - the best parameter vector of the last generation with shape (num_params)
        '''
        
        mean = old_run['mean']
        cov = old_run['cov']
        run_nbr = old_run['run_nbr']

        # continuing an old run?
        if mean:
            CMA.mean = mean
            CMA.covariance = cov
            CMA.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance=cov)
            CMA.model_dir = CMA.model_dir[:-1] + str(run_nbr)

        best_candidate = CMA.train_until_convergence(writer, model_dir)
        
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
    ctrl_layers = config.ctrl_layers
    stop_crit = config.stop_crit
    stop_median_crit = config.stop_median_crit
    num_parallel_agents = config.num_parallel_agents
    env_id = 'CarRacing-v0'

    ####
    # DEPRECATED
    ####
    # selection_pressure = config.selection_pressure
    #learning_rate = config.learning_rate
    #epochs = config.epochs
    #pop_size = config.pop_size
    ####
    ####

    if torch.cuda.is_available():
        device = 'cuda:0'
        ctrl_device = 'cuda:0'
    else:
        device = 'cpu'
        ctrl_device = 'cpu'

    # setting up paths
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = cur_dir + config.model_dir
    id_str = 'ctrl_results_better_rollouts'
    
    # init tensorboard
    log_dir = model_dir+id_str+'/'
    print(log_dir)
    (_,dirs,_) = os.walk(log_dir).__next__()
    run_id = 'run_' + str(len(dirs))
    log_dir = log_dir + run_id
    writer = SummaryWriter(log_dir)
    print('Saving results and logs in directory', log_dir)

    print('Setting up world model..')
    # set up visual model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    vis_model = modules.VAE(encoder, decoder, encode_only=True).to(device)
    
    # load visual model
    vis_model_file = 'better_variational_visual_epochs_1/lr_0.0036481/run_0/model.pt'
    vis_model.load_state_dict(torch.load(model_dir + vis_model_file, map_location=torch.device(device)))
    vis_model.eval()
    
    # load mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params).to(device)
    mdn_model_file = 'beter_mdnrnn_epochs_20/lr_0.001/temp_0.7/run_0/model.pt'
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

    dummy_model = modules.Controller(**ctrl_kwargs)
    nbr_params = count_parameters(dummy_model)

    # Wrapper around fitness function to ensure compatibility with CMA_ES API
    fit_func_kwargs = {
        'ctrl_kwargs':ctrl_kwargs,
        'model_class':modules.Controller,
        'env_id':env_id,
        'ctrl_device':ctrl_device,
        'obs_device':device,
        'vis_model':vis_model,
        'mdn_model':mdn_model
    }

    fitness_func = functools.partial(fitness, **fit_func_kwargs)

    CMA_parameters = {
        'nbr_params':nbr_params,
        'fitness_func':fitness_func,
        'stop_fitness':stop_crit,
        'stop_median_fitness':stop_median_crit
    }

    CMA = CMA_ES(**CMA_parameters)
    
    print('Starting training...')
    # init parallel processing
    if num_parallel_agents > 1:
        ray.init(num_cpus=num_parallel_agents, 
                object_store_memory=1024*1024*1024*num_parallel_agents,
                redis_max_memory=1024*1024*200
        )

    # train
    mean = None
    cov = None
    run_nbr = None
    old_run = {'mean':mean,'cov':cov,'run_nbr':run_nbr}
    best_parameters = train_CMA(CMA=CMA, writer=writer, stop_crit=stop_crit, old_run=old_run, model_dir=log_dir)
    
    print('Saving final model')
    # save model
    #load params
    best_model = modules.Controller(**ctrl_kwargs)
    best_model = set_params(best_model, best_parameters)
    
    torch.save(best_model.state_dict(), log_dir + '/controller_{}.pt'.format(int(time())))

    print('Showcasing the best model...')
    showcase(vis_model, mdn_model, best_model, env_id, device, ctrl_device)


    
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
    #parser.add_argument('--pop_size', type=int, default=1500, help='Population size for CMA-ES')
    parser.add_argument('--num_parallel_agents', type=int, default=10, help='Number of agents run in parallel when evaluating fitness')
    #parser.add_argument('--selection_pressure', type=float, default=0.7, help='Percentage of population that survives each iteration')
    parser.add_argument('--stop_crit', type=float, default=0.002, help='Average fitness value that needs to be reached')
    parser.add_argument('--stop_median_crit', type=float, default=0.005, help='Median fitness value that needs to be reached')
    #parser.add_argument('--batch_size', type=int, default=256, help='Number of examples to process in a batch')
    #parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    #parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    train(config)
