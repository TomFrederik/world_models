"""
In this script we train the visual world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader


def train(config):

    # get arguments
    input_dim = config.input_dim
    conv_layers = config.conv_layers
    z_dim = config.latent_dim
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    train_steps = config.train_steps
    device = torch.device(config.device)


    # set up model
    encoder = models.Encoder(input_dim, conv_layers, z_dim)
    # not sure if I somehow have to change dimensions of the conv layers here
    decoder = models.Decoder(input_dim, conv_layers, z_dim)
    model = models.VAE(encoder, decoder)

    # init crit and optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)


    # get data
    data = 
    batches = 

    # loss array
    losses = []
    
    for step, batch_input in enumerate(batches):

        # set grad to zero
        optimizer.zero_grad()

        # forward pass
        batch_output = model.forward(batch_input)

        # compute loss
        loss = criterion(batch_output, batch_input)

        # backward pass
        loss.backward()

        # updating weights
        optimizer.step()

        # log the losses
        losses.append(loss.item())

        if step % 100 == 0:
            print("Train step {}/{} , Loss = {}".format(step, train_steps, loss.item()))

    print('Done training.')
################################################################################
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_dim', type=tuple, default=(96,96,3), help='Dimensionality of input picture')
    parser.add_argument('--conv_layers', type=int, default=[[100, 3]], help='List of Conv Layers in the format [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--latent_dim', type=int, default=32, help="Dimension of the latent space")

    config = parser.parse_args()

    # Train the model
    train(config)