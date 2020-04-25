'''
inspect how well the variational autoencoder works
'''

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# other ptyhon modules
import argparse
from time import time
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt

# my modules
import models


def main(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    z_dim = config.latent_dim
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    cur_dir = os.path.dirname(os.path.realpath(__file__))

    model_dir = cur_dir + '/models/'
    data_dir = cur_dir + '/data/'

    (_,_,model_files) = os.walk(model_dir).__next__()
    (_,_,data_files) = os.walk(data_dir).__next__()

    # set up model
    encoder = models.Encoder(input_dim, conv_layers, z_dim)
    # not sure if I somehow have to change dimensions of the conv layers here
    decoder = models.Decoder(input_dim, conv_layers, z_dim)
    model = models.VAE(encoder, decoder).to(device)
    
    # load model
    model.load_state_dict(torch.load(model_dir + model_files[0]))

    # load data
    data = np.load(data_dir + data_files[1], allow_pickle=True)

    # get one image
    image = data[0,1,0]
    input_img = torch.from_numpy(np.reshape(image, (1,3,96,96))).to(device)
    input_img = input_img.float()
    out_image = model(input_img).cpu().detach().numpy()
    out_image = np.reshape(out_image, (96,96,3))

    # plot images
    plt.figure(1)
    plt.imshow(image)
    plt.savefig('input_img.pdf')

    plt.figure(2)
    plt.imshow(out_image)
    plt.savefig('output_img.pdf')
    

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(3,96,96), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[10, 3], [10,3]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")
    parser.add_argument('--model_dir', type=str, default='/models/', help="Relative directory for saving models")
    

    config = parser.parse_args()

    # Train the model
    main(config)