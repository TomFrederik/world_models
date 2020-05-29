'''
Script to define the CMA-ES model.
'''

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

# gym modules
import gym
from gym import wrappers

# ray for parallization
import ray

# other python packages
import numpy as np
from time import time
import os


# matrix square root
from sqrtm import sqrtm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CMA_ES:

    def __init__(self, nbr_params, fitness_func, stop_fitness=0.01, stop_median_fitness=0.01, pop_size=None, mu=None, weights=None, sigma=0.5, mean=None):
        '''
        params:
        nbr_params - int, n in the algorithm, dimension of the search space
        fitness_func - function, fitness function to minimize. Should take a population of vectors x and returns a float
        stop_fitness - float, if at least one x has better fitniss than this, training is halted
        pop_size - int, lambda in the algorithm
        mu - int, selected number per generation
        weights - list or 1D ndarray, recombination weights for the mu selected vectors
        sigma - float, initial step size for CMA
        mean - ndarray, initial mean for CMA
        '''
        self.pop_size = pop_size
        self.nbr_params = nbr_params
        self.mu = mu
        self.weights = weights
        self.fitness_func = fitness_func
        self.stop_fitness = stop_fitness
        self.stop_median_fitness = stop_median_fitness
        self.sigma = sigma
        self.mean = mean

        # calculate default params if they are not overridden
        if pop_size == None:
            self.pop_size = int(4 + np.floor(3*np.log(nbr_params)))

        if mean == None:
            self.mean = torch.zeros(nbr_params)

        if mu == None:
            self.mu = self.pop_size // 2

        if weights == None:
            self.weights = []
            for i in range(1, self.pop_size+1):
                self.weights.append(np.log((self.pop_size + 1) / 2) - np.log(i))
        self.weights = torch.from_numpy(np.array(self.weights))

        self.mu_eff = torch.sum(self.weights[:self.mu]) ** 2 / torch.sum(self.weights[:self.mu] ** 2)
        self.mu_eff_bar = torch.sum(self.weights[self.mu:]) ** 2 / torch.sum(self.weights[self.mu:] ** 2)
        self.mu_w = (torch.sum(self.weights[:mu] ** 2)) ** -1 

        self.c_c = (4 + self.mu_eff / nbr_params) / (nbr_params + 4 + 2 * self.mu_eff / nbr_params)
        self.c_1 = 2 / ((nbr_params + 1.3) ** 2 +  self.mu_eff)
        self.c_mu = np.min([1-self.c_1, 2 * ( (self.mu_eff - 2 + 1/self.mu_eff) / ((nbr_params + 2) ** 2 + self.mu_eff) )])
        self.c_sig = (self.mu_eff + 2) / (nbr_params + self.mu_eff + 5)
        self.d_sig = 1 + 2 * np.max([0, np.sqrt((self.mu_eff -1) / (nbr_params + 1) ) - 1]) + self.c_sig

        self.alpha_mu_bar = 1 + self.c_1 / self.c_mu
        self.alpha_mu_eff_bar = 1 + 2 * self.mu_eff_bar / (self.mu_eff + 2)
        self.alpha_posdef_bar = (1 - self.c_1 - self.c_mu) / (nbr_params * self.c_mu)

        self.weights[self.weights >= 0] = self.weights[self.weights >= 0] / torch.sum(self.weights[self.weights > 0])
        self.weights[self.weights < 0] = self.weights[self.weights < 0] * np.min([self.alpha_mu_bar, self.alpha_mu_eff_bar, self.alpha_posdef_bar]) / torch.sum(-1 * self.weights[self.weights < 0])

        self.p_c = torch.zeros(nbr_params)
        self.p_sig = torch.zeros(nbr_params)

        self.cov = torch.eye(nbr_params)
        self.sqrt_inv_cov = torch.eye(nbr_params)

    def train_until_convergence(self, writer, model_dir):
        '''
        Executes CMA-ES until convergence is reached, with the parameters specified in the init function
        params:
        writer - tensorboard logger
        model_dir - str, directory to save the training results in
        '''

        converged = False
        ctr = 0
        
        while not converged:

            # sample set of vectors
            self.cur_pop = self.get_sample()

            # evaluate on fitness function

            # start counter
            self.total_start_time = time()

            self.fitness = self.fitness_func(self.cur_pop)

            # rank pop by fitness
            idcs = torch.argsort(self.fitness)
            self.fitness = self.fitness[idcs]
            self.cur_pop = self.cur_pop[idcs,:]

            mean_fitness = torch.mean(self.fitness)
            best_fitness = self.fitness[0]
            median_fitness = self.fitness[int(len(self.fitness)//2)]
            
            writer.add_scalar('mean fitness', mean_fitness, ctr)
            writer.add_scalar('best fitness', best_fitness, ctr)
            writer.add_scalar('median fitness', median_fitness, ctr)
            writer.add_scalar('det covariance', torch.det(self.cov), ctr)
            writer.add_scalar('entropy', self.dist.entropy().item(), ctr)
            writer.flush()

            print('Just completed step {0:5d}, average fitness of last step was {1:4.3f}'.format(ctr+1, mean_fitness))
            print('Saving best candidate..')
            best_candidate = self.cur_pop[0]
            torch.save(best_candidate, f=model_dir+'/best_candidate.pt')
            print('Saving covariance and mean..')
            torch.save(self.mean, f=model_dir+'/mean.pt')
            torch.save(self.cov, f=model_dir+'/cov.pt')

            ctr += 1

            # test convergence criteria
            converged = self.convergence_test()
            if converged:
                continue

            # update mean
            self.old_mean = self.mean.clone()
            self.mean = torch.mean(self.cur_pop[:self.mu], dim=0)

            # update p_sig
            self.p_sig = self.get_new_p_sig()

            # update p_c
            self.p_c = self.get_new_p_c()

            # update C
            self.cov = self.get_new_C()
            self.sqrt_inv_cov = sqrtm(self.cov.inverse()) # torch has no C^(-1/2) natively 
            
            # update sigma
            self.sigma = self.get_new_sig()

        print('Convergence criterium reached!')
        print('The best achieved fitness in the last generation was {0:1.4f}'.format(self.fitness[0]))
        print('The median fitness of the last generation was {0:1.4f}'.format(self.fitness[int(len(self.fitness)//2)]))
        return self.cur_pop[0]
    
    def get_new_p_sig(self):
        '''
        Compute new evo path for sigma
        '''
        p_sig = (1 - self.c_sig) * self.p_sig + np.sqrt(1 - (1 - self.c_sig)**2) * np.sqrt(self.mu_w) * self.sqrt_inv_cov * (self.mean - self.old_mean) / self.sigma

        return p_sig    

    def get_new_p_c(self):
        '''
        Compute new evo path for covariance matrix
        '''
        # compute indicator function
        self.norm_sig = np.sqrt(torch.sum(self.p_sig ** 2))
        if self.norm_sig >= 0 and self.norm_sig <= 1.5 * np.sqrt(self.nbr_params):
            self.indic = 1
        else:
            self.indic = 0
        
        # update p_c
        p_c = (1 - self.c_c) * self.p_c + self.indic * np.sqrt(1 - (1 - self.c_c)**2) * np.sqrt(self.mu_w) * (self.mean - self.old_mean) / self.sigma 
        
        return p_c

    def get_new_C(self):
        '''
        Update covariance matrix
        '''
        self.c_s = (1 - self.indic * self.norm_sig ** 2) * self.c_1 * self.c_c * (2 - self.c_c)
        old_term = (1 - self.c_1 - self.c_mu + self.c_s) * self.cov

        rank_one = self.c_1 * torch.ger(self.p_c, self.p_c)

        rank_mu = torch.zeros((self.nbr_params, self.nbr_params))
        scaled_diff = (self.cur_pop[:self.mu] - self.old_mean) / self.sigma

        for i in range(self.mu):
            rank_mu += self.weights[i] * torch.ger(scaled_diff[i], scaled_diff[i])
        rank_mu *= self.c_mu

        next_cov = old_term + rank_one + rank_mu

        return next_cov

    def get_new_sig(self):
        '''
        Update step size sigma
        '''
        sig = self.sigma * np.exp(self.c_sig / self.d_sig * ( torch.sum(self.p_sig ** 2) / (np.sqrt(self.nbr_params) - 0.25 / np.sqrt(self.nbr_params)) - 1))
        
        return sig
    
    def get_sample(self):
        '''
        Samples pop_size params from N(self.mean, self.cov)
        '''

        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.mean, covariance_matrix=self.cov)

        sample = self.dist.rsample(sample_shape=torch.Size([self.pop_size]))

        return sample
        
    def convergence_test(self):
        '''
        Test wether convergence is reached.
        Convergence is defined as best_fitness <= stop_fitness and median_fitness <= stop_median_fitness
        '''
        if self.fitness[0] <= self.stop_fitness and self.fitness[int(len(self.fitness)//2)] <= self.stop_median_fitness:
            return True
        else:
            return False


class CMA_ES_old:
    
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
        #print('center shape',centering.shape) # [900,900]
        # calc mean
        self.mean = torch.mean(pop, dim=0)
        #print(mean.shape) # [867]
        # calc cov matrix as 1/(n-1) M^T M, where M is X-mean(X)
        diff_matrix = torch.matmul(centering, pop)
        #print('diff_matrix.shape', diff_matrix.shape) # [900,867]
        self.covariance = 1/(pop.shape[0]-1) * torch.matmul(diff_matrix.t(), diff_matrix)
        #print(covariance)
        #print('covariance.shape',covariance.shape) # [867,867]
        # create new dist object
        new_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.mean, covariance_matrix=self.covariance)

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
    
    @ray.remote(num_gpus=0.0625)
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
            print('\n')
            print('###################')
            print('\n\n')
            print('Finished rollout with id {}. Duration: {} minutes and {} seconds'.format(id, duration//60, duration%60))
            print('Total time elapsed since start of this generation: {} minutes and {} seconds'.format(total_duration//60, total_duration%60))
            print('\n\n')
            print('###################')
            print('\n')
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
        
        # list of all the models instantiated with their respective parameters
        models = [self.model_class(**self.model_kwargs) for _ in range(self.pop_size)]
        for run_id in range(self.pop_size):
            #load params
            models[run_id].layers[-1].weight.data = torch.reshape(pop[run_id,:864], (3,288))
            models[run_id].layers[-1].bias.data = torch.reshape(pop[run_id,864:], torch.Size([3]))
        
        # run each agent once
        cum_rew_ids = [self.run_agent.remote(self, model, id) for (model, id) in zip(models, np.arange(self.pop_size))]
        
        # save as fitness and return
        for run_id in range(self.pop_size):
            fitness[run_id] = ray.get(cum_rew_ids[run_id])
                
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
