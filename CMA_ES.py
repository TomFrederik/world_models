'''
Script to define the CMA-ES model.
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

# ray for parallization
import ray

# other python packages
import numpy as np
from time import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CMA_ES:
    
    def __init__(self, model_class, vis_model, mdn_rnn, model_dir, ctrl_device='cpu', model_kwargs={}, env_id='CarRacing-v0', num_parallel_agents=4, pop_size=1000, selection_pressure=0.1):

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
        self.model_dir = model_dir

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
                print('Saving best candidate..')
                best_candidate = cur_pop_fitness[torch.argsort(cur_pop_fitness, descending=True)[0]]
                torch.save(best_candidate, f=self.model_dir+'best_candidate.pt')
        
        best_candidate = cur_pop_fitness[torch.argsort(cur_pop_fitness, descending=True)[0]]
        print('Completed training after {0:5d} steps. Best fitness of last step was {1:4.3f}'.format(ctr, best_candidate))
        print('Saving best candidate..')
        best_candidate = cur_pop_fitness[torch.argsort(cur_pop_fitness, descending=True)[0]]
        torch.save(best_candidate, f=self.model_dir+'best_candidate.pt')
        
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
        new_pop = dist.sample(torch.Size([self.pop_size]))

        return new_pop, dist
    
    @ray.remote(num_gpus=0.125)
    def run_agent(self, model, id):
        #print(next(self.mdn_rnn.parameters()).is_cuda)
        with torch.no_grad():
            env = gym.make(self.env_id)
            done = False
            cum_rew = 0
            step = 1
            start_time = time()

            # put model on cuda
            # needs to happen inside this function otherwise model is on cpu again, Cthulu knows why
            model.to(self.ctrl_device)

            # get first obs
            obs = env.reset()
            obs = np.reshape(obs, (1,3,96,96))
            obs = torch.from_numpy(obs).to(self.obs_device)
            obs = obs.float() / 255

            # first pass through world model
            vis_out, _ = self.vis_model(obs) 
            torch.cuda.empty_cache()
            mdn_hidden = self.mdn_rnn.intial_state(batch_size=vis_out.shape[0]).to(self.obs_device)
            mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
            
            # first pass through controller
            ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)
            torch.cuda.empty_cache()

            action = model(ctrl_in)
            action = torch.squeeze(action)
            
            while not done:
                
                obs, rew, done, _ = env.step(action.cpu().detach().numpy())
                cum_rew += rew

                '''
                if step % 100 == 0:
                    print('Steps completed in this run: ',step)
                    duration = time()-start_time
                    print('Time since start: {} minutes and {} seconds.'.format(duration//60,duration%60))
                '''

                obs = np.reshape(obs, (1,3,96,96))
                obs = torch.from_numpy(obs).to(self.obs_device)
                obs = obs.float() / 255

                # pass through world model
                vis_out, _ = self.vis_model(obs)

                torch.cuda.empty_cache()
                mdn_in = torch.unsqueeze(torch.cat([vis_out, torch.unsqueeze(action, dim=0)], dim=1), dim=1)
                mdn_hidden = self.mdn_rnn.forward(mdn_in, h_0=torch.unsqueeze(mdn_hidden, dim=0))
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(self.ctrl_device)

                # first pass through controller
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)

                torch.cuda.empty_cache()
                action = model(ctrl_in)
                torch.cuda.empty_cache()
                action = torch.squeeze(action)
                
                # inc step counter
                step += 1

            env.close()            
            duration = time()-start_time
            total_duration = time() - self.total_start_time
            print('###################')
            print('###################')
            print('\n\n\n')
            print('Finished rollout with id {}. Duration: {} minutes and {} seconds'.format(id, duration//60, duration%60))
            print('Total time elapsed since start of this generation: {} minutes and {} seconds'.format(total_duration//60, total_duration%60))
            print('\n\n\n')
            print('###################')
            print('###################')
            return cum_rew

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

        # start counter
        self.total_start_time = time()

        # calc number of parallel runs
        if self.pop_size % self.num_parallel_agents != 0:
            raise ValueError('Pop size needs to be divisible by number of parallel agents, but are {} and {} respectively.'.format(self.pop_size, self.num_parallel_agents))
        num_runs = self.pop_size // self.num_parallel_agents

        start_time = time()

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

                vis_out, _ = self.vis_model(obs)
                #print(vis_out.shape) # [1,32]

                mdn_hidden = self.mdn_rnn.intial_state(batch_size=vis_out.shape[0]).to(self.obs_device)
                #print(mdn_hidden.shape) # [1,1,256]
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
                #print(mdn_hidden.shape) # [1,256]
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(self.ctrl_device)
                #print(ctrl_in.shape) # [1,288]
                action = model(ctrl_in) 
                #print(action.shape) # [1,3]
                run_acs[:,0,:] = action
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
                    
                    mdn_in = torch.unsqueeze(torch.cat([vis_out, torch.unsqueeze(action, dim=0)], dim=1), dim=1)
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
            cum_rew = []
            models = [self.model_class(**self.model_kwargs) for _ in range(self.pop_size)]
            #print(next(models[0].parameters()).is_cuda)
            #raise NotImplementedError
            for run_id in range(self.pop_size):
                #load params
                models[run_id].layers[-1].weight.data = torch.reshape(pop[run_id,:864], (3,288))
                models[run_id].layers[-1].bias.data = torch.reshape(pop[run_id,864:], torch.Size([3]))
            
            cum_rew_ids = [self.run_agent.remote(self, model, id) for (model, id) in zip(models, np.arange(self.pop_size))]
            for run_id in range(self.pop_size):
                cum_rew.append(ray.get(cum_rew_ids[run_id]))
            fitness = torch.from_numpy(np.array(cum_rew))
                    
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
