"""
This script samples 10,000 random rollouts on the carracing environment
"""

import gym
from gym import wrappers
import numpy as np
import argparse

def main(id, number_rollouts):
    env = gym.make('CarRacing-v0')
    env = wrappers.Monitor(env, "./gym_results",force=True)
   
    rollouts = []
    counter = 1

    for _ in range(number_rollouts):

        done = False # reset done condition
        obs = env.reset() # reset environment and save inital observation

        # all actions and observations for one rollout
        actions = []
        observations = []

        while not done:
            # one rollout

            # take random action
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act) 

            # save action and observation
            actions.append(act)
            observations.append(obs)
        
        # save rollout
        rollouts.append((actions, observations))
        
        counter += 1
        
        # save progress so far
        if counter%100 == 0:
            np.save('random_rollouts_{}.npy'.format(id), rollouts)
    
    env.close()

    np.save('random_rollouts_{}.npy'.format(id), rollouts)

if __name__ == "__main__":
    
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--number_rollouts', type=int, default=1000)

    config = parser.parse_args()
    
    main(config.id, config.number_rollouts)
