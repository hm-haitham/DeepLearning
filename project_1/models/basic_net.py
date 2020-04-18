import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.model_name = config.BASIC_NET_NAME
        
        self.hidden = nn.Linear(config.DOUBLE_IMAGE_SIZE, config.BASIC_NET_HIDDEN_LAYER)
        
        stdv = 1. / math.sqrt(self.hidden.weight.size(1)) 
        
        self.hidden.weight.data.uniform_(-stdv, stdv) 
        if self.hidden.bias is not None: 
            self.hidden.bias.data.uniform_(-stdv, stdv)
        
        self.output = nn.Linear(config.BASIC_NET_HIDDEN_LAYER, 1)

    def forward(self, x):
        flattened = x.view(x.size(0),-1)
        
        hid = self.hidden(flattened)
        hid = F.relu(hid)
        
        out = self.output(hid)
        
        return torch.sigmoid(out)