import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class BasicNet(nn.Module):

    def __init__(self, nb_hidden_layers = config.BASIC_NET_NB_HIDDEN, hidden_layer = config.BASIC_NET_HIDDEN_LAYER):
        super(BasicNet, self).__init__()
        
        if nb_hidden_layers < 1:
            raise Exception("Minimum 1 hidden layers for basic net" )
        
        #list of nb_hidden_layers-1 Linear layers (hidden_layer,hidden_layer)
        self.hiddens = nn.ModuleList([nn.Sequential(nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)) for i in range(nb_hidden_layers-1)])
        
        #Linear layer (2*14*14, hidden_layer)
        self.hiddens.insert(0, nn.Linear(config.DOUBLE_IMAGE_SIZE, hidden_layer))
        
        self.output = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        
        flattened = x.view(x.size(0),-1)
        
        hid = flattened
        
        for layer in self.hiddens:
            hid = layer(hid)
        
        out = self.output(hid)
        
        return torch.sigmoid(out)