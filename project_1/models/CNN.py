import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class CNN(nn.Module):

    def __init__(self, nb_hidden_layers = config.CNN_NB_HIDDEN, base_channel_size = config.CNN_BASE_CHANNEL_SIZE, hidden_layer = config.CNN_HIDDEN_LAYER, kernel_size = config.CNN_KERNEL_SIZE):
        
        super(CNN, self).__init__()
        
        if nb_hidden_layers < 1:
             raise Exception("Minimum 1 hidden layers for the CNN ")
        
        self.base_channel_size = base_channel_size
     
        conv_channel_size = self.base_channel_size*2
        
        #(W−F+2P)/S+1
        #k= 1, 3, 5  
        self.conv_net = nn.Sequential(nn.Conv2d(1, self.base_channel_size, kernel_size= kernel_size ),  #(14-k)+1 = 15 -k # 14, 12,10
                                                 nn.LeakyReLU(),
                                                 nn.MaxPool2d(kernel_size=2, stride=2), #(15-k) / 2 : 7, 6, 5  
                                                 nn.Dropout(p=0.2),
                                                 nn.Conv2d(self.base_channel_size, conv_channel_size, 
                                                           kernel_size = kernel_size),    #7, 5, 3   # (15 - k) /2 - k + 1
                                                 nn.LeakyReLU(),
                                                 nn.Dropout(p=0.2) )

        fcn_input_size = int( ( ((15 - kernel_size) /2 - kernel_size + 1)** 2 ) * conv_channel_size)
        
        
        self.fc_net = nn.ModuleList([nn.Sequential(nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)) for i in range(nb_hidden_layers-1)])

        self.fc_net.insert(0,nn.Sequential(nn.Linear(fcn_input_size, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)))

        self.output = nn.Linear(hidden_layer, config.NUMBER_OF_CLASSES)
            

    def forward(self, x):
        conved = self.conv_net(x)
        
        flattened = conved.view(conved.size(0),-1)
        
        hid = flattened
        
        for block in self.fc_net:
            hid = block(hid)
        
        out = self.output(hid)
        
        return F.softmax(out, dim=1), out