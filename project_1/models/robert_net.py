import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class RobertNet(nn.Module):

    def __init__(self, nb_hidden_layers, base_channel_size = config.ROBERT_NET_BASE_CHANNEL_SIZE, hidden_layer = config.ROBERT_NET_HIDDEN_LAYER):
        
        super(RobertNet, self).__init__()
        self.model_name = config.ROBERT_NET_NAME
        
        if nb_hidden_layers < 0:
             raise Exception("Minimum 0 hidden layers for " + self.model_name)
        
        self.base_channel_size = base_channel_size
        
        conv_channel_size = self.base_channel_size*2
        
        
        #Change to Module list instead of Sequential if the number of ConvNets is dynamic (i.e passed as parameter)
        self.conv_net = nn.Sequential(nn.Conv2d(1, self.base_channel_size, kernel_size=3),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(self.base_channel_size, conv_channel_size, kernel_size=3),
                                      nn.LeakyReLU(),
                                      nn.MaxPool2d(2))
        
        
        #We can make the network more general by not hardcoding the "4" and computing instead
        #Basically if we want specify the padding in convLayers, the number of the convlayers or the kernel sizes (3 parameters)
        #We compute what is the height or width at the end of the conv_net
        #Because in this kind of network we Pool one time (more would make us lose a lot of information as 14x14 is small
        #The "2" can stay hardcoded, however careful to check that the computed height/width at the end of the convnet
        #after tweaking the kernel size / padding and other can still be divisble by 2. Otherwise there will be a runtime error
        #when training it.
        width_height_conv = ( config.WIDTH_HEIGHT - 4 ) / 2
        
        fcn_input_size = int(width_height_conv * width_height_conv * conv_channel_size)
        
        self.fc_net = nn.ModuleList()
        
        if nb_hidden_layers > 0:
            self.fc_net = nn.ModuleList([nn.Linear(hidden_layer, hidden_layer) for i in range(nb_hidden_layers-1)])

            self.fc_net.insert(0,nn.Linear(fcn_input_size, hidden_layer))

            self.output = nn.Linear(hidden_layer, config.NUMBER_OF_CLASSES)
            
        if nb_hidden_layers == 0:
            self.output = nn.Linear(fcn_input_size, config.NUMBER_OF_CLASSES)

    def forward(self, x):
        conved = self.conv_net(x)
        
        flattened = conved.view(conved.size(0),-1)
        
        hid = flattened
        
        for layer in self.fc_net:
            hid = layer(hid)
            hid = F.relu(hid)
        
        out = self.output(hid)
        
        return F.softmax(out, dim=1), out