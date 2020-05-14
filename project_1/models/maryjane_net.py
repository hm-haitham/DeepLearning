import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class MaryJaneNet(nn.Module):

    def __init__(self, nb_hidden_layers = config.MARYJANE_NET_NB_HIDDEN, base_channel_size = config.MARYJANE_NET_BASE_CHANNEL_SIZE, hidden_layer = config.MARYJANE_NET_HIDDEN_LAYER, kernel_size = config.MARYJANE_NET_KERNEL_SIZE):
        
        super(MaryJaneNet, self).__init__()
        self.model_name = config.MARYJANE_NET_NAME
        
        if nb_hidden_layers < 0:
             raise Exception("Minimum 0 hidden layers for " + self.model_name)
        
        self.base_channel_size = base_channel_size
        
        conv_channel_size = self.base_channel_size*2
        
        #Change to Module list instead of Sequential if the number of ConvNets is dynamic (i.e passed as parameter)
        #(W−F+2P)/S+1
        #k= 1, 3, 5  
        self.conv_net = nn.Sequential(nn.Conv2d(1, self.base_channel_size, kernel_size= kernel_size ),  #(14-k)+1 = 15 -k # 14, 12,10
                                                 nn.LeakyReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2), #(15-k) / 2 : 7, 6, 5  
                                                 nn.Dropout(p=0.2),
                                                 nn.Conv2d(self.base_channel_size, conv_channel_size, 
                                                           kernel_size = kernel_size),    #7, 5, 3   # (15 - k) /2 - k + 1
                                                 nn.LeakyReLU(),
                                                 #nn.MaxPool2d(kernel_size=3, stride=3), #1  
                                                 nn.Dropout(p=0.2) )

        fcn_input_size = int( ( ((15 - kernel_size) /2 - kernel_size + 1)** 2 ) * conv_channel_size)
        
        self.fc_net = nn.ModuleList()
        
        if nb_hidden_layers > 0:
            self.fc_net = nn.ModuleList([nn.Sequential(nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)) for i in range(nb_hidden_layers-1)])

            self.fc_net.insert(0,nn.Sequential(nn.Linear(fcn_input_size, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)))

            self.output = nn.Linear(hidden_layer, config.NUMBER_OF_CLASSES)
            
        if nb_hidden_layers == 0:
            self.output = nn.Linear(fcn_input_size, config.NUMBER_OF_CLASSES)

    def forward(self, x):
        conved = self.conv_net(x)
        
        flattened = conved.view(conved.size(0),-1)
        
        hid = flattened
        
        for block in self.fc_net:
            hid = block(hid)
        
        out = self.output(hid)
        
        return F.softmax(out, dim=1), out