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

# gym modules
import gym
from gym import wrappers

# stable baselines
from stable_baselines.common import make_vec_env


# other python packages
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CMA_ES:
    
    def __init__(self, model_class, vis_model, mdn_rnn, model_kwargs={}, env_id='CarRacing-v0', num_runs=5, pop_size=20, selection_pressure=0.1):

        # save params in class instance
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.env_id = env_id
        self.num_runs = num_runs
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
        cur_pop_fitness = self.fitness(cur_pop)

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

        # calc mean
        mean = torch.mean(pop, dim=0)
        # calc cov matrix as 1/(n-1) M^T M, where M is X-mean(X)
        diff_matrix = pop - mean.repeat(pop.shape[0],1)
        covariance = 1/(pop.shape[0]-1) * torch.matmul(torch.transpose(diff_matrix), diff_matrix)

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

        # compute new distribution
        self.dist = self.get_new_dist(survivors)

        # sample from new dist
        new_pop = self.sample(self.pop_size)

        return new_pop, self.dist


        
    def fitness(self, pop):
        '''
        params:
        pop - a population of parameter vectors with shape (pop_size, num_params)
        
        returns:
        fitness - fitness values of each parameter vector. Is of shape (pop_size)
        '''

        # make env
        env = gym.make(self.env_id)

        # container for fitness values
        fitness = torch.zeros(self.pop_size)

    
        for agent_id in range(self.pop_size):
            print('Evaluating {}th agent'.format(agent_id+1))
            agent_params = pop[agent_id,:]
            
            model = self.model_class(**self.model_kwargs)

            # set model params to this agent's params
            model.layers[-1].weight.data = torch.reshape(agent_params[:864], (3,288))
            model.layers[-1].bias.data = torch.reshape(agent_params[864:], torch.Size([3]))
            
            # create environment
            vec_env = make_vec_env(self.env_id, n_envs=self.num_runs)
            vec_obs = vec_env.reset()
            print(vec_obs)
            raise NotImplementedError
            
            for i in range(self.num_runs):
                # collect rollouts
                obs = env.reset()
                obs = np.reshape(obs, (3,96,96))
                obs = torch.from_numpy(obs).to(self.obs_device)
                obs = torch.unsqueeze(obs, dim=0)
                obs = obs.type(torch.cuda.FloatTensor) / 255

                run_zs = torch.zeros((1,1000,32))
                run_acs = torch.zeros((1,1000,3))

                vis_out, _ = self.vis_model(obs)
                #print(vis_out.shape) # [1,32]
                run_zs[:,0,:] = vis_out
                mdn_hidden = self.mdn_rnn.intial_state(batch_size=1).to(self.obs_device)
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to('cpu')
                action = model(ctrl_in)
                action = torch.squeeze(action)

                done = False
                cum_rew = 0
                step = 1
                while not done:
                    step += 1
                    obs, rew, done, _ = env.step(action.detach().numpy())
                    obs = np.reshape(obs, (3,96,96))
                    obs = torch.from_numpy(obs).to(self.obs_device)
                    obs = torch.unsqueeze(obs, dim=0)
                    obs = obs.type(torch.cuda.FloatTensor) / 255

                    
                    vis_out,_ = self.vis_model(obs)
                
                    mdn_hidden = self.mdn_rnn(torch.cat([run_zs[:,:step,:], run_acs[:,:step,:]], dim=2))  
                    mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(self.obs_device)
                    ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to('cpu')
                    
                    action = model(ctrl_in)
                    action = torch.squeeze(action)
                    cum_rew += rew
                print('Agent {} achieved {} reward on its {}th run.'.format(agent_id+1, cum_rew, i+1))
                fitness[agent_id] += cum_rew / self.num_runs


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
