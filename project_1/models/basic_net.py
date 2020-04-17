import torch.nn as nn
import torch
from torch.nn import functional as F

class BasicNet(nn.Module):

    def __init__(self):
        super(BasicNet, self).__init__()
        self.model_name = "basic_net"
        
        self.hidden = nn.Linear(784, 1400)
        
        self.output = nn.Linear(1400, 1)

    def forward(self, x):
        hid = self.hidden(x)
        
        out = self.output(hid)
        
        return torch.sigmoid(out)