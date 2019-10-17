""" 
In this file I define the neural networks for the 
implementation of the world models paper
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, layers, z_dim):
        """
        params:
        input_dim: expects a three dimensional array (x_dim, y_dim, channel)
        batch_size: expects an integer
        layers: list of conv layers = [[out_0, kernel_size_0], [out_1, kernel_size_1], ...]
        z_dim: the dimension of the latent space
        """

        # init nn.Module
        super().__init__()

        
        self.input_dim = input_dim

        self.layers = []
        
        # set up chain of Convolutions
        prev_layer = [len(input_dim), 0]

        for _, layer in enumerate(layers):
        
            self.layers.append(nn.Conv3d(prev_layer[0], layer[0], layer[1]))
            prev_layer = layer
        
        # final layer are dense layers to compute mean and variance of 
        # multivariate gaussian
        self.mu = nn.Linear(layers[-1][0], z_dim, bias = True)
        self.var = nn.Linear(layers[-1][0], z_dim, bias = True)
    
        # initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform(layer.weight)
        nn.init.xavier_uniform(self.mu.weight)
        nn.init.xavier_uniform(self.var.weight)


        # convert to torch list
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """
        implements forward pass

        params:
        x: expects something that is of shape [batch_size, input_dim]
        
        returns:
        parameterization of the latent space
        """

        if x.shape[1:] != self.input_dim:
            print("Input is of dimension {}, expected {}".format(x.shape[1:], self.input_dim))
            raise ValueError

        
        # pass x through all convolutions
        # activation is relu
        for i, layer in enumerate(self.layers):
            if i == 0: 
                hidden = F.relu(self.layers[0](x))
                continue
            hidden = F.relu(layer(hidden))
        
        # compute mean and variance of z:
        z_mean = self.mu(hidden)
        z_var = self.var(hidden)

        return z_mean, z_var


class Decoder(nn.Module):

    def __init__(self, input_dim, layers, z_dim):
        """
        params:
        input_dim: expects a three dimensional array (x_dim, y_dim, channel)
        batch_size: expects an integer
        layers: list of conv layers = [[in_0, kernel_size_0], [in_1, kernel_size_1], ...]
        z_dim: the dimension of the latent space
        """

        # init nn.Module
        super().__init__()

        # save for later
        self.z_dim = z_dim
        self.input_dim = input_dim

        # init list of layers
        self.layers = []
        
        # set up chain of Convolutions
        next_layer = [len(input_dim), 0]

        for _, layer in reversed(list(enumerate(layers))):
            
            self.layers.append(nn.ConvTranspose3d(layer[0], next_layer[0], layer[1]))
            next_layer = layer
        
        # reverse list so you can pass through it the correct order during forward pass
        self.layers.reverse()

        # first layer is a dense layer to convert latent space 
        self.linear = nn.Linear(z_dim, layers[0][0], bias=True)
    
        # initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform(layer.weight)
        nn.init.xavier_uniform(self.linear.weight)

        # convert to torch list
        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        """
        implements forward pass

        params:
        x: expects something that is of shape [batch_size, z_dim]
        
        returns:
        
        """

        if x.shape[1:] != self.z_dim:
            print("Input is of dimension {}, expected {}".format(x.shape[1:], self.z_dim))
            raise ValueError

        
        # pass x through linear layer
        # activation is relu
        hidden = F.relu(self.linear(x))

        # pass hidden through all convolutions
        for i, layer in enumerate(self.layers):
            hidden = F.relu(layer(hidden))


        out = hidden
        
        return out


class VAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        
        # encoding step

        z_mean, z_var = self.encoder(x)

        # sample from this distribution
        # with reparameterization trick
        std = torch.sqrt(z_var) # use some other reparameterization?
        eps = torch.randn_like(std)
        # z = eps * std + mean
        z_sample = eps.mul(std).add_(z_mean) 

        # decoding step
        prediction = self.decoder(z_sample)

        return prediction