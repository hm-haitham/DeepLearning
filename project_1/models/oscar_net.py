import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class OscarNet(nn.Module):

    def __init__(self, nb_hidden_layers, hidden_layer = config.OSCAR_NET_HIDDEN_LAYER):
        super(OscarNet, self).__init__()
        self.model_name = config.OSCAR_NET_NAME
        
        if nb_hidden_layers < 1:
            raise Exception("Minimum 1 hidden layers for " + self.model_name)
        
        self.hiddens = nn.ModuleList([nn.Linear(hidden_layer, hidden_layer) for i in range(nb_hidden_layers-1)])
        
        self.hiddens.insert(0, nn.Linear(config.SINGLE_IMAGE_SIZE, hidden_layer))
        
        self.output = nn.Linear(hidden_layer, config.NUMBER_OF_CLASSES)

    def forward(self, x):
        flattened = x.view(x.size(0),-1)
        
        hid = flattened
        
        for layer in self.hiddens:
            hid = layer(hid)
            hid = F.relu(hid)
        
        out = self.output(hid)
        
        return F.softmax(out, dim=1), out