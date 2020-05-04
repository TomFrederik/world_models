'''
Script to define the CMA-ES model
'''

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# parallel processing
import torch.multiprocessing as mp

# gym modules
import gym
from gym import wrappers

# stable baselines
#from stable_baselines.common import make_vec_env

# other python packages
import numpy as np
from time import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CMA_ES:
    
    def __init__(self, model_class, vis_model, mdn_rnn, ctrl_device='cpu', model_kwargs={}, env_id='CarRacing-v0', num_parallel_agents=4, pop_size=1000, selection_pressure=0.1):

        # save params in class instance
        self.ctrl_device = ctrl_device
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.env_id = env_id
        self.num_parallel_agents = num_parallel_agents
        self.pop_size = pop_size
        self.selection_pressure = selection_pressure
        self.vis_model = vis_model
        self.mdn_rnn = mdn_rnn

        if next(vis_model.parameters()).is_cuda:
            self.obs_device = 'cuda:0'
        else:
            self.obs_device = 'cpu'

        # get number of params
        dummy_model = model_class(**model_kwargs)
        self.num_params = count_parameters(dummy_model)
        del dummy_model

        # get init normal distributions
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(self.num_params), covariance_matrix=torch.eye(self.num_params, self.num_params))

    def train(self, stop_crit=600):
        '''
        Executes the CMA-ES algorithm.
        params:
        stop_crit - average fitness of the population that triggers stopping training

        returns:
        best_candidate - the best parameter vector of the last generation with shape (num_params)
        '''

        # sample initial population
        cur_pop = self.sample(self.pop_size)

        # calc fitness of current pop
        with torch.no_grad():
            cur_pop_fitness = self.fitness(cur_pop)
        print(cur_pop_fitness)

        done = False
        ctr = 0
        while not done:

            # compute new dist and sample for new pop
            cur_pop, self.dist = self.evolution_step(cur_pop, cur_pop_fitness)

            # calc fitness of current pop
            cur_pop_fitness = self.fitness(cur_pop)

            # check if done
            if torch.mean(cur_pop_fitness) >= stop_crit:
                done = True

            ctr += 1

            if ctr % 1 == 0:
                print('Just completed step {0:5d}, average fitness of last step was {1:4.3f}'.format(ctr, torch.mean(cur_pop_fitness)))
        
        best_candidate = cur_pop_fitness[torch.argsort(cur_pop_fitness, descending=True)[0]]
        print('Completed training after {0:5d} steps. Best fitness of last step was {1:4.3f}'.format(ctr, best_candidate))

        return best_candidate

    def get_new_dist(self, pop):
        '''
        params:
        pop - parameter matrix with shape (num_survivors, num_parameters)

        returns:
        new_dist - torch Multivariate normal dist with mean and covariance matrix calculated from the population
        '''

        assert pop.shape == (int(self.pop_size * self.selection_pressure), self.num_params), 'Pop shape is of different shape than expected: {}'.format(pop.shape)
        #print(pop)
        #print(pop.shape)
        
        centering = torch.eye(pop.shape[0]) - 1/(pop.shape[0]) * torch.ones((pop.shape[0], pop.shape[0]))
        print('center shape',centering.shape)
        # calc mean
        mean = torch.mean(pop, dim=0)
        #print(mean.shape) # [867]
        # calc cov matrix as 1/(n-1) M^T M, where M is X-mean(X)
        diff_matrix = torch.matmul(centering, pop)
        print('diff_matrix.shape', diff_matrix.shape)
        covariance = 1/(pop.shape[0]-1) * torch.matmul(diff_matrix.t(), diff_matrix)
        #print(covariance)
        print('covariance.shape',covariance.shape) # [867,867]
        # create new dist object
        new_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=covariance)

        return new_dist

    def evolution_step(self, cur_pop, cur_pop_fitness):
        '''
        params:
        cur_pop - parameter matrix with shape (pop_size, num_params)
        cur_pop_fitness - fitness values of current pop, shape (pop_size)

        returns:
        new_pop - parameter matrix with shape (pop_size, num_params)
        new_dist - new distribution based on the best parameters of the new population
        '''
        # calc survivors
        survivors = self.grim_reaper(cur_pop, cur_pop_fitness, self.selection_pressure)
        #print(survivors)
        #print(survivors.shape)

        # compute new distribution
        dist = self.get_new_dist(survivors)

        # sample from new dist
        new_pop = dist.sample(self.pop_size)

        return new_pop, dist

    def parallel_initial_action(self, model, ctrl_in, id):
        '''
        takes an action for a single agent
        this function will be run in parallel for different models
        '''
        
        action = model(ctrl_in)
        #print(action.shape) # [1,3]
        action = torch.squeeze(action).detach().numpy()
        #print(action.shape) # [3]

        return [action, id]
    
    def parallel_action(self, model, ctrl_in, id):
        '''
        takes an action for a single agent
        this function will be run in parallel for different models
        '''
        
        action = model(ctrl_in)
        action = torch.squeeze(action).detach().numpy()
        
        return [action, id]

    def fitness(self, pop):
        '''
        params:
        pop - a population of parameter vectors with shape (pop_size, num_params)
        
        returns:
        fitness - fitness values of each parameter vector. Is of shape (pop_size)
        '''
        '''
        # make env
        env = gym.make(self.env_id)
        '''
        

        # container for fitness values
        fitness = torch.zeros(self.pop_size)

        # calc number of parallel runs
        if self.pop_size % self.num_parallel_agents != 0:
            raise ValueError('Pop size needs to be divisible by number of parallel agents, but are {} and {} respectively.'.format(self.pop_size, self.num_parallel_agents))
        num_runs = self.pop_size // self.num_parallel_agents

        start_time = time()

        # don't know how to properly parallelize with different parameters for each agent
        if self.num_parallel_agents == 1:
            for run_id in range(num_runs):
                print('Evaluating agent no. {}'.format(run_id+1))
                agent_params = pop[run_id,:]
                
                model = self.model_class(**self.model_kwargs)

                # set model params to this agent's params
                model.layers[-1].weight.data = torch.reshape(agent_params[:864], (3,288))
                model.layers[-1].bias.data = torch.reshape(agent_params[864:], torch.Size([3]))
                
                # create environment
                env = gym.make(self.env_id)
                obs = env.reset()
                obs = np.reshape(obs, (1,3,96,96))
                obs = torch.from_numpy(obs).to(self.obs_device).detach()
                obs = obs.float() / 255
                #print(obs.shape) # [1,3,96,96]
                self.run_zs = torch.zeros((1,1000,32))
                self.run_acs = torch.zeros((1,1000,3))

                vis_out, _ = self.vis_model(obs)
                #print(vis_out.shape) # [1,32]
                self.run_zs[:,0,:] = vis_out
                mdn_hidden = self.mdn_rnn.intial_state(batch_size=vis_out.shape[0]).to(self.obs_device)
                #print(mdn_hidden.shape) # [1,1,256]
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
                #print(mdn_hidden.shape) # [1,256]
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)
                #print(ctrl_in.shape) # [1,288]
                action = model(ctrl_in) 
                #print(action.shape) # [1,3]
                action = torch.squeeze(action)
                #print(action.shape) # [3]

                done = np.array([False] * self.num_parallel_agents)
                cum_rew = np.zeros(self.num_parallel_agents)
                step = 1
                for i in range(999): # one run is 1000 steps.. somehow done is not true but new tracks are automatically generated after 1000 steps
                    if step % 100 == 0:
                        print('Steps completed in this run: ',step)
                        duration = time()-start_time
                        print('Time since start: {} minutes and {} seconds.'.format(duration//60,duration%60))
                    obs, rew, done, _ = env.step(action.detach().numpy())
                    
                    #print(obs.shape) # [1,96,96,3]
                    #print(rew.shape) # [1]
                    #print(done.shape) # [1]
                    obs = np.reshape(obs, (1,3,96,96))
                    obs = torch.from_numpy(obs).to(self.obs_device).detach()
                    obs = obs.float()/ 255

                    vis_out,_ = self.vis_model(obs)
                    mdn_in = torch.unsqueeze(torch.cat([self.run_zs[:,step,:], self.run_acs[:,step,:]], dim=1), dim=1)
                    mdn_hidden = self.mdn_rnn.forward(mdn_in, h_0=torch.unsqueeze(mdn_hidden, dim=0))
                    mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(self.obs_device)
                    ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)
                    
                    action = model(ctrl_in)
                    action = torch.squeeze(action)
                    
                    cum_rew += rew
                    step += 1

                mean_rew = np.mean(cum_rew)
                print('Agent {} achieved an average reward of {}.'.format(run_id+1, mean_rew))
                fitness[run_id] = mean_rew
                env.close()
                
        
        else:
            for run_id in range(num_runs):
                print('Evaluating agents no. {} to {}'.format(run_id+1, run_id+self.num_parallel_agents))
                agent_params = pop[run_id:run_id+self.num_parallel_agents,:]
                
                models = [self.model_class(**self.model_kwargs) for _ in range(self.num_parallel_agents)]

                # set model params to this agent's params
                for i in range(len(models)):
                    models[i].layers[-1].weight.data = torch.reshape(agent_params[i,:864], (3,288))
                    models[i].layers[-1].bias.data = torch.reshape(agent_params[i,864:], torch.Size([3]))
                
                # create environment
                env = make_vec_env(self.env_id, n_envs=self.num_parallel_agents)
                obs = env.reset()
                obs = np.reshape(obs, (obs.shape[0],3,96,96))
                obs = torch.from_numpy(obs).to(self.obs_device).detach()
                obs = obs.float() / 255
                #print(obs.shape) # [4,3,96,96]

                # create storage arrays for encodings and actions
                self.run_zs = torch.zeros((obs.shape[0],1000,32))
                self.run_acs = torch.zeros((obs.shape[0],1000,3))

                # first pass through world model
                vis_out, _ = self.vis_model(obs)
                #print(vis_out.shape) # [4,32]
                self.run_zs[:,0,:] = vis_out
                mdn_hidden = self.mdn_rnn.intial_state(batch_size=vis_out.shape[0]).to(self.obs_device)
                #print(mdn_hidden.shape) # [1,4,256]
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
                #print(mdn_hidden.shape) # [4,256]
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)
                #print(ctrl_in.shape) # [4,288]
                
                # take parallel actions
                pool = mp.Pool(self.num_parallel_agents)
                ids = np.arange(0,self.num_parallel_agents)
                args = list(zip(models, ctrl_in, ids))
                pool_out = [pool.apply_async(self.parallel_initial_action, args=arg).get() for arg in args]
                pool.close()
                pool.join()
                action = np.zeros((self.num_parallel_agents, 3))
                for i in range(self.num_parallel_agents):
                    action[pool_out[i][1]] = pool_out[i][0]
                
                #done = np.array([False] * self.num_parallel_agents)
                cum_rew = np.zeros(self.num_parallel_agents)
                step = 1
                for i in range(999): # one run is 1000 steps.. somehow done is not true but new tracks are automatically generated after 1000 steps
                    if step % 100 == 0:
                        print('Steps completed in this run: ',step)
                        duration = time()-start_time
                        print('Time since start: {} minutes and {} seconds.'.format(duration//60,duration%60))
                    env.step_async(action)
                    obs, rew, done, _ = env.step_wait()
                    #print(obs.shape) # [4,96,96,3]
                    #print(rew.shape) # [4]
                    #print(done.shape) # [4]
                    obs = np.reshape(obs, (obs.shape[0],3,96,96))
                    obs = torch.from_numpy(obs).to(self.obs_device).detach()
                    obs = obs.float() / 255

                    #pass through world model    
                    vis_out,_ = self.vis_model(obs)
                    mdn_in = torch.unsqueeze(torch.cat([self.run_zs[:,step,:], self.run_acs[:,step,:]], dim=1), dim=1)
                    mdn_hidden = self.mdn_rnn.forward(mdn_in, h_0=torch.unsqueeze(mdn_hidden, dim=0))
                    mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(self.obs_device)
                    ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)

                    # take parallel actions
                    pool = mp.Pool(self.num_parallel_agents)
                    ids = np.arange(0,self.num_parallel_agents)
                    args = list(zip(models, ctrl_in, ids))
                    pool_out = [pool.apply_async(self.parallel_action, args=arg).get() for arg in args]
                    pool.close()
                    pool.join()
                    action = np.zeros((self.num_parallel_agents, 3))
                    for i in range(self.num_parallel_agents):
                        action[pool_out[i][1]] = pool_out[i][0]
                    
                    cum_rew += rew
                    step += 1
                print(cum_rew)
                print(cum_rew.shape)
                mean_rew = np.mean(cum_rew)
                print('Agents no. {} to {} achieved an average reward of {}.'.format(run_id+1, run_id+self.num_parallel_agents, mean_rew))
                fitness[run_id:run_id+self.num_parallel_agents] = cum_rew
                env.close()
            
        return fitness

    def sample(self, pop_size):
        '''
        params:
        pop_size - number of parameter vectors that should be sampled

        returns:
        sample - parameter matrix of shape (pop_size, num_params)
        '''

        sample = self.dist.rsample(sample_shape=torch.Size([pop_size]))

        return sample

    def grim_reaper(self, cur_pop, cur_pop_fitness, selection_pressure):
        '''
        params:
        cur_pop - parameter matrix of shape (pop_size, n)
        selection_pressure - percentage of population that is allowed to survive 
        cur_pop_fitness - fitness values of pop, shape (pop_size)

        returns:
        survivors - parameter matrix of shape (num_survivors, n)
        '''

        # calculate number of survivors
        num_survivors = int(cur_pop.shape[0] * selection_pressure)

        # get IDs of best agents
        survivor_ids = torch.argsort(cur_pop_fitness, descending=True)[:num_survivors]

        # return best agents
        survivors = cur_pop[survivor_ids,:]

        return survivors
