"""
This script samples 500 random rollouts on the car racing environment
"""

import gym
from gym import wrappers
import numpy as np
import argparse
from time import time
import os

def main(config):

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = cur_dir + config.data_path
    
    env = gym.make('CarRacing-v0')

    rollouts = []

    for counter in range(config.nbr_rollouts):

        # reset rollout array to free up space
        if counter > 0 and counter % config.set_size == 0:
            id = time()
            np.save(data_dir + 'random_rollouts_{}.npy'.format(id), np.array(rollouts), allow_pickle=True)
            rollouts = []

        start_time = time()

        done = False # reset done condition
        obs = env.reset() # reset environment and save inital observation

        # all actions and observations for one rollout
        actions = []
        observations = []

        iter = 1
        while not done:
            # one rollout
            #print('Step number {} in rollout {}'.format(iter, counter+1))

            # take random action
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act) 

            # save action and observation
            actions.append(act)
            observations.append(obs)
            iter += 1

        # append rollout
        rollouts.append(np.array([actions, observations]))
    
        counter += 1
        print('Generated {} tracks so far'.format(counter))
        print('This track took {} seconds to simulate.'.format(time()-start_time))

    if len(rollouts)>0:
        id = time()
        np.save(data_dir + 'random_rollouts_{}.npy'.format(id), np.array(rollouts), allow_pickle=True)

    env.close()



if __name__ == "__main__":
 
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--nbr_rollouts', type=int, default=1000)
    parser.add_argument('--data_path', type=str, default='/data/' )
    parser.add_argument('--set_size', type=int, default=100)

    config = parser.parse_args()
    
    main(config)
