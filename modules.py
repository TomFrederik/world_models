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
        z_var =  torch.exp(self.var(hidden))
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

    def __init__(self, encoder, decoder, encode_only=False):
        super().__init__()

        self.encode_only = encode_only

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

        # get KL loss
        kl_loss = -0.5 - torch.log(std) + (std**2 + z_mean**2)/2
        kl_loss = torch.sum(kl_loss) # sum over whole batch at once
        
        if self.encode_only:
            return z_sample, kl_loss
        else:
            # decoding step
            prediction = self.decoder(z_sample)

            return prediction, kl_loss 

class MDN(nn.Module):

    def __init__(self, input_dim=256+32, layers=[100,100], nbr_gauss=5, temp=1, out_dim=32):

        if temp<=0:
            raise ValueError('temperature parameter needs to be larger than 0, but is {}'.format(temp))

        super(MDN, self).__init__()

        self.temp = temp
        self.nbr_gauss = nbr_gauss
        self.out_dim = out_dim

        self.layers = []

        self.layers.append(nn.Linear(in_features=input_dim, out_features=layers[0]))
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i+1]))
        self.layers.append(nn.Linear(in_features=layers[-1], out_features=2*self.nbr_gauss+self.nbr_gauss*self.out_dim)) # (2 + out_dim) * nbr_gauss

        self.layers = nn.ModuleList(self.layers)

        # initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)


    def forward(self, input):

        if input.is_cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'

        #print('MDN Input')
        #print(input.shape)

        # pass through linear layers
        hidden = input
        for i in range(len(self.layers)-1):
            hidden = F.relu(self.layers[i].forward(hidden))
        hidden = self.layers[-1](hidden)

        # predict gaussians with temperature modulation
        coeff  = F.softmax(hidden[:,:self.nbr_gauss] / self.temp, dim=1) # mixture coefficients
        var  = torch.exp(hidden[:,self.nbr_gauss:2*self.nbr_gauss]) * self.temp # variances
        #print('var 1')
        #print(var)
        '''
        dummy = torch.zeros((var.shape[0],self.nbr_gauss, self.nbr_gauss))
        for i in range(var.shape[0]):
            dummy[i] = torch.diag(var[i])
        var = dummy
        del dummy
        #print('var 2')
        #print(var)
        '''
        mean  = hidden[:,2*self.nbr_gauss:] # means
        mean = torch.reshape(mean, (mean.shape[0], self.nbr_gauss, mean.shape[1]//self.nbr_gauss)) # e.g. (10,5,32)
        
        return coeff, mean, var

class MDN_RNN(nn.Module):
    
    def __init__(self, input_dim=32+3, lstm_units=256, lstm_layers=1, nbr_gauss=5, mdn_layers=[100,100], temp=1):

        if temp<=0:
            raise ValueError('temperature parameter needs to be larger than 0, but is {}'.format(temp))

        super(MDN_RNN, self).__init__()

        self.nbr_gauss = nbr_gauss

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_units, num_layers=lstm_layers, batch_first=True)

        # input is [h_t, a_t, z_t], output is z_t
        self.mdn = MDN(input_dim=lstm_units, layers=mdn_layers, nbr_gauss=nbr_gauss, temp=temp, out_dim=input_dim-3)



    def forward(self, input):

        #print('MDN_RNN Input')
        # print(input.shape) # should be (batch_size, seq_len, z_dim+3), checks out

        if input.is_cuda:
            device = 'cuda:0'
        else:
            device = 'cpu'

        coeff_preds = torch.zeros((input.shape[0], input.shape[1]-1, self.nbr_gauss)).to(device)
        mean_preds = torch.zeros((input.shape[0], input.shape[1]-1, self.nbr_gauss, input.shape[-1]-3)).to(device)
        var_preds = torch.zeros((input.shape[0], input.shape[1]-1, self.nbr_gauss)).to(device)

        for t in range(input.shape[1]-1):
            if t == 0:
                out_t, (h_t, c_t) = self.lstm(torch.unsqueeze(input[:,t,:], dim=1))
            else:
                out_t, (h_t, c_t) = self.lstm(torch.unsqueeze(input[:,t,:], dim=1), (h_t, c_t))
            
            # predict distribution
            coeff_pred, mean_pred, var_pred = self.mdn(torch.squeeze(h_t))

            # save distribution
            coeff_preds[:,t,:] = coeff_pred
            mean_preds[:,t,:,:] = mean_pred
            var_preds[:,t,:] = var_pred

        return coeff_preds, mean_preds, var_preds


class Controller(nn.Module):

    def __init__(self, in_dim=256+32, layers=[], ac_dim=3):

        super(Controller, self).__init__()

        if len(layers) == 0:
            self.layers = [nn.Linear(in_dim, ac_dim)]
        else:
            prev_layer = in_dim
            self.layers = []
            for i in range(len(layers)):
                self.layers.append(nn.Linear(prev_layer, layers[i]))
                prev_layer = layers[i]
            self.layers.append(prev_layer, ac_dim)
        
        self.layers = nn.ModuleList(self.layers)
    
        # initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)


    def forward(self, input):

        hidden = input

        for layer in self.layers[-1]:
            hidden = F.relu(layer(hidden))
        out = F.tanh(self.layers[-1](hidden))

        return out