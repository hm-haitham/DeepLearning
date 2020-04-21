import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class OscarNet(nn.Module):

    def __init__(self, nb_hidden_layers):
        super(OscarNet, self).__init__()
        self.model_name = config.OSCAR_NET_NAME
        
        if nb_hidden_layers < 1 :
            raise Exception("Minimum 1 hidden layer")
        
        self.hiddens = []
        
        self.hiddens.append(nn.Linear(config.SINGLE_IMAGE_SIZE, config.OSCAR_NET_HIDDEN_LAYER))
        
        stdv = 1. / math.sqrt(self.hiddens[0].weight.size(1)) 
        self.hiddens[0].weight.data.uniform_(-stdv, stdv) 
        if self.hiddens[0].bias is not None: 
            self.hiddens[0].bias.data.uniform_(-stdv, stdv)
        
        for i in range(nb_hidden_layers):
            self.hiddens.append(nn.Linear(config.OSCAR_NET_HIDDEN_LAYER, config.OSCAR_NET_HIDDEN_LAYER))
            
            stdv = 1. / math.sqrt(self.hiddens[i].weight.size(1)) 
            self.hiddens[i].weight.data.uniform_(-stdv, stdv) 
            if self.hiddens[i].bias is not None: 
                self.hiddens[i].bias.data.uniform_(-stdv, stdv)
        
        self.output = nn.Linear(config.OSCAR_NET_HIDDEN_LAYER, config.NUMBER_OF_CLASSES)

    def forward(self, x):
        flattened = x.view(1,-1)
        
        hid = self.hiddens[0](flattened)
        
        for i in range(1, len(self.hiddens)):
            hid = self.hiddens[i](hid)
            hid = F.relu(hid)
        
        out = self.output(hid)
        
        return torch.sigmoid(out)