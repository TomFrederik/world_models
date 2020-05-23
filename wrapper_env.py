import gym
from gym import error, spaces, utils
from gym.utils import seeding

import torch
import numpy as np

class WrapperEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, env_id, vis_model, mdn_model):

    super(WrapperEnv, self).__init__()
    self.env = gym.make(env_id)

    if next(vis_model.parameters()).is_cuda:
        self.obs_device = 'cuda:0'
    else:
        self.obs_device = 'cpu'    
    
    self.action_space = self.env.action_space
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(32+256,)) 

    self.vis_model = vis_model
    self.mdn_model = mdn_model

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    
    obs = np.reshape(obs, (1,3,96,96))
    obs = torch.from_numpy(obs).to(self.obs_device)
    obs = obs.float() / 255

    # pass through world model
    vis_out, _ = self.vis_model(obs) 
    mdn_hidden = self.mdn_model.intial_state(batch_size=vis_out.shape[0]).to(self.obs_device)
    mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
    
    obs = torch.flatten(torch.cat((vis_out, mdn_hidden), dim=1)).cpu().detach().numpy()
   
    self.done = done
    
    return obs, rew, done, info

  def reset(self):
    obs = self.env.reset()
    obs = np.reshape(obs, (1,3,96,96))
    obs = torch.from_numpy(obs).to(self.obs_device)
    obs = obs.float() / 255

    # pass through world model
    vis_out, _ = self.vis_model(obs) 
    mdn_hidden = self.mdn_model.intial_state(batch_size=vis_out.shape[0]).to(self.obs_device)
    mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
    
    obs = torch.flatten(torch.cat((vis_out, mdn_hidden), dim=1)).cpu().detach().numpy()

    return obs

  def render(self, mode='human'):
    self.env.render(mode=mode)

  def close(self):
    self.env.close()
    