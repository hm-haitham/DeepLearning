import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class DesmondNet(nn.Module):

    def __init__(self, nb_hidden_layers, left_net, right_net):
        super(DesmondNet, self).__init__()
        self.model_name = config.DESMOND_NET_NAME
        
        if nb_hidden_layers < 1 :
            raise Exception("Minimum 1 hidden layer")
        
        self.left_net = left_net
        
        self.right_net = right_net
        
        self.hiddens = []
        
        self.hiddens.append(nn.Linear(config.NUMBER_OF_CLASSES*2, config.DESMOND_NET_HIDDEN_LAYER))
        
        stdv = 1. / math.sqrt(self.hiddens[0].weight.size(1)) 
        self.hiddens[0].weight.data.uniform_(-stdv, stdv) 
        if self.hiddens[0].bias is not None: 
            self.hiddens[0].bias.data.uniform_(-stdv, stdv)
        
        for i in range(nb_hidden_layers):
            self.hiddens.append(nn.Linear(config.DESMOND_NET_HIDDEN_LAYER, config.DESMOND_NET_HIDDEN_LAYER))
            
            stdv = 1. / math.sqrt(self.hiddens[i].weight.size(1)) 
            self.hiddens[i].weight.data.uniform_(-stdv, stdv) 
            if self.hiddens[i].bias is not None: 
                self.hiddens[i].bias.data.uniform_(-stdv, stdv)
        
        self.output = nn.Linear(config.DESMOND_NET_HIDDEN_LAYER, 1)

    def forward(self, x):
        #TODO: SPLIT x which is of size [N, 2, 14, 14] to two distinct tensors of size [N, 1, 14, 14]
        dummy = torch.zeros(x.size(0), 1, 14, 14)
        left_image = dummy
        right_image = dummy
        
        lefted = self.left_net(dummy)
        
        righted = self.right_net(dummy)
        
        #TODO: CONCAT lefted and righted which are of size [N,10] each to a single tensor of size [N,20]
        dummy_inter = torch.zeros(x.size(0), righted.size(1)*2)
        hid = dummy_inter
        
        hid = self.hiddens[0](hid)
        
        for i in range(1, len(self.hiddens)):
            hid = self.hiddens[i](hid)
            hid = F.relu(hid)
        
        out = self.output(hid)
        
        return torch.sigmoid(out)