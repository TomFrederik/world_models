"""
Learning to work with the gym environment
"""


import gym

"""
env = gym.make('MountainCar-v0')
env.reset() # returns an initial observation
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


# get info, obs, done and reward after each timestep
env = gym.make('CartPole-v0')
for i_episodes in range(20): 
    obs = env.reset() # get initial observation
    for t in range(100):
        env.render() 
        print(obs) # array of four floats
        action = env.action_space.sample() # sample a random action
        obs, rew, done, info = env.step(action)
        print(info) # empty dict
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close
"""

"""
# "What are Spaces?"
env = gym.make('CartPole-v0')
print(env.action_space)
# Discrete (2) = fixed range of non-negative numbers, here 0 and 1
print(env.observation_space)
# Box(4,) = 4-dim box, so here 4 float numbers in an array, that are bounded
print(env.observation_space.high)
print(env.observation_space.low)
# get lower and upper bound for the observation space

# sample from a space 
space = env.action_space
x = space.sample()
# is x in space?
assert space.contains(x)
# how large is space?
print('size of space is', space.n)
env.reset()
for _ in range(100):
    env.render()
    env.step(0) # 0 = left, 1 = right
env.close()
"""

"""
# what environments are there?
print("You have access to the following environments:")
print(gym.envs.registry.all())
"""

# looking at carracing
env = gym.make('CarRacing-v0')
obs = env.reset()
for i in range(1000):
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample())
env.close()