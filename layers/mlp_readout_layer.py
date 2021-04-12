import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lahelr import lahelr_print

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):
        '''
        L=nb_hidden_layers

        '''
        super().__init__()
        lahelr_print("when init mlp readout layer: input and output dim are:",input_dim,output_dim) #4 6 in:52
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ] # half the size every layer
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        '''
        x-> Linears(half the size every layer, total L layers, splited by relu)->Linear(to the output-dim)->
        '''
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y