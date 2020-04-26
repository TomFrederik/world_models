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
        prev_layer = [input_dim[0], 0]

        for _, layer in enumerate(layers):
        
            self.layers.append(nn.Conv2d(prev_layer[0], layer[0], layer[1], stride=2))
            prev_layer = layer
        

        # final layer are dense layers to compute mean and variance of 
        # multivariate gaussian
        self.mu = nn.Linear(4096, z_dim, bias = True)
        self.var = nn.Linear(4096, z_dim, bias = True)
    
        # initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.var.weight)


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

        if tuple(list(x.shape[1:]))  != self.input_dim:
            print("Input is of dimension {}, expected {}".format(tuple(list(x.shape[1:])) , self.input__dim))
            raise ValueError
        
        # pass x through all convolutions
        # activation is relu
        for i, layer in enumerate(self.layers):
            if i == 0: 
                hidden = F.relu(self.layers[0](x))
                continue
            hidden = F.relu(layer(hidden))

        # flatten
        hidden = torch.flatten(hidden, start_dim=1)
        #print(hidden)
        # compute mean and variance of z:
        z_mean = self.mu(hidden)
        z_var =  F.relu(self.var(hidden))
        #print(z_mean)
        #print(z_var)

        return z_mean, z_var


class Decoder(nn.Module):

    def __init__(self, input_dim, layers, z_dim):
        """
        params:
        input_dim: expects a three dimensional array (x_dim, y_dim, channel)
        layers: list of deconv layers = [[in_0, kernel_size_0], [in_1, kernel_size_1], ...]
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
        prev_layer = [4096, 0]

        for _, layer in enumerate(layers):
            self.layers.append(nn.ConvTranspose2d(prev_layer[0], layer[0], layer[1], stride=2))
            prev_layer = layer

        # first layer is a dense layer to convert latent space 
        self.linear = nn.Linear(z_dim, 4096, bias=True)
    
        # initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.linear.weight)

        # convert to torch list
        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        """
        implements forward pass

        params:
        x: expects something that is of shape [batch_size, z_dim]
        
        returns:
        
        """

        if list(x.shape[1:])[0]  != self.z_dim:
            print("Input is of dimension {}, expected {}".format(list(x.shape[1:])[0] , self.z_dim))
            raise ValueError

        #print(x)

        # pass x through linear layer
        hidden = self.linear(x)

        # reshape
        hidden = torch.reshape(hidden, (hidden.shape[0], 4096, 1, 1))

        # pass hidden through all convolutions
        for i, layer in enumerate(self.layers):
            hidden = F.relu(layer(hidden))

        out = torch.sigmoid(hidden)
        #print(out)
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
        eps = torch.randn_like(std) # samples from N(mu=0,std=1)
        z_sample = eps.mul(std).add_(z_mean) # z = eps * std + mean

        # decoding step
        prediction = self.decoder(z_sample)

        return prediction


class MDN_RNN(nn.Module):
    
    def __init__(self, input_dim, lstm_units=256, lstm_layers=1, nbr_gauss=5):

        super(MDN_RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_units, num_layers=lstm_layers)



    def forward(input):

        pass