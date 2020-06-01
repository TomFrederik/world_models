import torch
import numpy as np

import modules

def showcase(vis_model, mdn_model, agent, env_id, obs_device, ctrl_device):
    ''' showcases performance of agent'''
    while True:
        with torch.no_grad():
            env = gym.make(env_id)
            done = False
            cum_rew = 0
            step = 1
            start_time = time()    
            
            obs = env.reset()
            obs = np.reshape(obs, (1,3,96,96))
            obs = torch.from_numpy(obs).to(obs_device)
            obs = obs.float() / 255
            
            # first pass through world model
            vis_out, _ = vis_model(obs) 
            mdn_hidden = mdn_model.intial_state(batch_size=vis_out.shape[0]).to(obs_device)
            mdn_hidden = torch.squeeze(mdn_hidden, dim =0)
            
            # first pass through controller
            ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(ctrl_device)

            action = agent(ctrl_in)
            action = torch.squeeze(action)

            while not done:
                
                obs, rew, done, _ = env.step(action.cpu().detach().numpy().astype(int))
                cum_rew += rew

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

                mdn_in = torch.unsqueeze(torch.cat([vis_out, torch.unsqueeze(action, dim=0)], dim=1), dim=1)
                mdn_hidden = mdn_model.forward(mdn_in, h_0=torch.unsqueeze(mdn_hidden, dim=0))
                mdn_hidden = torch.squeeze(mdn_hidden, dim =0).to(ctrl_device)

                # first pass through controller
                ctrl_in = torch.cat((vis_out,mdn_hidden), dim=1).to(ctrl_device)

                torch.cuda.empty_cache()
                action = agent(ctrl_in)
                action = torch.squeeze(action)
                
                # inc step counter
                step += 1

            env.close()


def main(config):
    print('Setting up world model..')
    # set up visual model
    encoder = modules.Encoder(input_dim, conv_layers, z_dim)
    decoder = modules.Decoder(input_dim, deconv_layers, z_dim)
    vis_model = modules.VAE(encoder, decoder, encode_only=True).to(device)
    
    # load visual model
    vis_model_file = 'better_variational_visual_epochs_1/lr_0.0036481/run_0/model.pt'
    vis_model.load_state_dict(torch.load(model_dir + vis_model_file, map_location=torch.device(device)))
    vis_model.eval()
    
    # load mdn model
    mdn_params = {'input_dim':z_dim+3, 'lstm_units':lstm_units, 'lstm_layers':lstm_layers, 'nbr_gauss':nbr_gauss, 'mdn_layers':mdn_layers, 'temp':temp}
    mdn_model = modules.MDN_RNN(**mdn_params).to(device)
    mdn_model_file = 'better_mdnrnn_epochs_20/lr_0.001/temp_1/run_0/model.pt'
    mdn_model.load_state_dict(torch.load(model_dir + mdn_model_file, map_location=torch.device(device)))
    mdn_model.eval()

    # set up CMA-ES
    # parameters for control network
    ctrl_kwargs = {
        'input_dim':lstm_units+z_dim,
        'layers':ctrl_layers,
        'ac_dim':3
    }
    agent = modules.Controller(**ctrl_kwargs)
    agent_file = 'ctrl_results_better_rollouts/start_var_1/run_0/model.pt'
    agent.load_state_dict(torch.load(model_dir + agent_file, map_location=torch.device(device)))
    agent.eval()

    showcase(vis_model, mdn_model, agent, env_id, device, ctrl_device)


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[32, 4], [64,4], [128,4], [256,4]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--deconv_layers', type=int, default=[[128, 4], [64,4], [32,4], [8,4], [3,6]], help='List of Deconv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of layers in the LSTM')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units per layer')
    parser.add_argument('--nbr_gauss', type=int, default=5, help='Number of gaussians for MDN')
    parser.add_argument('--mdn_layers', type=int, default=[100,100,50,50], help='List of layers in the MDN')
    parser.add_argument('--temp', type=float, default=1, help='Temperature for mixture model')
    parser.add_argument('--ctrl_layers', type=int, default=[], help='List of layers in the Control network')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    
    config = parser.parse_args()

    # Train the model
    main(config)