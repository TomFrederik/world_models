"""
This script samples 500 random rollouts on the carracing environment
"""

import gym
from gym import wrappers
import numpy as np
import argparse

def main(id, number_rollouts):
    env = gym.make('CarRacing-v0')

    rollouts = []

    for counter in range(number_rollouts):

        done = False # reset done condition
        obs = env.reset() # reset environment and save inital observation

        # all actions and observations for one rollout
        actions = []
        observations = []

        iter = 1
        while not done:
            # one rollout
            print('Step number {} in rollout {}'.format(iter, counter+1))

            # take random action
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act) 

            # save action and observation
            actions.append(act)
            observations.append(obs)
            iter += 1

        # save rollout
        rollouts.append(np.array([actions, observations]))
        
        
        # save progress so far
        if (counter + 1) % 100 == 0:
            np.save('/data/random_rollouts_{}_{}.npy'.format(id, counter+1), rollouts)
    
        counter += 1

    env.close()

    try:
        np.save('/data/random_rollouts_{}_{}.npy'.format(id, counter+1), rollouts, allow_pickle=True)
    except:
        raise ValueError('Save didnt work!')


if __name__ == "__main__":
 
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--number_rollouts', type=int, default=1000)

    config = parser.parse_args()
    
    main(config.id, config.number_rollouts)
