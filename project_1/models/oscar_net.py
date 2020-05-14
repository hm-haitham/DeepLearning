import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class OscarNet(nn.Module):

    def __init__(self, nb_hidden_layers= config.OSCAR_NET_NB_HIDDEN, hidden_layer = config.OSCAR_NET_HIDDEN_LAYER):
        super(OscarNet, self).__init__()
        self.model_name = config.OSCAR_NET_NAME
        
        if nb_hidden_layers < 1:
            raise Exception("Minimum 1 hidden layers for " + self.model_name)
        
        self.hiddens = nn.ModuleList([nn.Sequential(nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)) for i in range(nb_hidden_layers-1)])

        self.hiddens.insert(0,nn.Sequential(nn.Linear(config.SINGLE_IMAGE_SIZE, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)))
        
        self.output = nn.Linear(hidden_layer, config.NUMBER_OF_CLASSES)

    def forward(self, x):
        flattened = x.view(x.size(0),-1)
        
        hid = flattened
        
        for block in self.hiddens:
            hid = block(hid)
        
        out = self.output(hid)
        
        return F.softmax(out, dim=1), out