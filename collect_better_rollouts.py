"""
This script samples 500 random rollouts on the car racing environment
"""

import gym
from gym import wrappers
import numpy as np
import argparse
from time import time
import os
import ray
import modules
import torch
from time import time

@ray.remote
def run_agent(model, id, env_id, ctrl_device, obs_device, total_start_time, vis_model, mdn_model):
    with torch.no_grad():
        env = gym.make(env_id)
        done = False
        cum_rew = 0
        observations = []
        actions = []
        step = 1
        start_time = time()    
        
        obs = env.reset()
        observations.append(obs)
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
        actions.append(action.numpy())

        while not done:
            
            obs, rew, done, _ = env.step(action.cpu().detach().numpy().astype(int))
            cum_rew += rew
            observations.append(obs)

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
            actions.append(action.numpy())

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
        return np.array([actions, observations])

def main(config):
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
    nbr_rollouts = config.nbr_rollouts
    env_id = 'CarRacing-v0'
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        ctrl_device = 'cuda:0'
    else:
        device = 'cpu'
        ctrl_device = 'cpu'

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    data_dir = cur_dir + config.data_path

    model_dir = cur_dir + '/models/'
    model_path = model_dir + 'ctrl_results/run_0/best_candidate.pt'
    
    # set up visual model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    vis_model = modules.VAE(encoder, decoder, encode_only=True).to(device)
    
    # load visual model
    vis_model_file = 'variational_visual_epochs_1_lr_0.001/1588429800.pt'
    vis_model.load_state_dict(torch.load(model_dir + vis_model_file, map_location=torch.device(device)))
    vis_model.eval()
    
    # load mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params).to(device)
    mdn_model_file = 'mdnrnn_epochs_20_lr_0.001_layers_100_100_50_50_schedsteps_100.pt'
    mdn_model.load_state_dict(torch.load(model_dir + mdn_model_file, map_location=torch.device(device)))
    mdn_model.eval()

    ctrl_kwargs = {
        'input_dim':lstm_units+z_dim,
        'layers':ctrl_layers,
        'ac_dim':3
    }

    ctrl_model = modules.Controller(**ctrl_kwargs)
    ctrl.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    env = gym.make(env_id)

    total_start_time = time()

    rollouts = [run_agent.remote(ctrl_model, id, env_id, ctrl_device, device, total_start_time, vis_model, mdn_model) for id in np.arange(nbr_rollouts)]

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
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[32, 4], [64,4], [128,4], [256,4]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--deconv_layers', type=int, default=[[128, 4], [64,4], [32,4], [8,4], [3,6]], help='List of Deconv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of layers in the LSTM')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units per layer')
    parser.add_argument('--nbr_gauss', type=int, default=5, help='Number of gaussians for MDN')
    parser.add_argument('--mdn_layers', type=int, default=[100,100,50,50], help='List of layers in the MDN')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for mixture model')
    parser.add_argument('--ctrl_layers', type=int, default=[], help='List of layers in the Control network')

    config = parser.parse_args()
    
    main(config)
