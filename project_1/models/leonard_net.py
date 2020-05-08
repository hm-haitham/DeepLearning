import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class LeonardNet(nn.Module):

    def __init__(self, nb_hidden_layers, image_net, hidden_layer=config.LEONARD_NET_HIDDEN_LAYER):
        super(LeonardNet, self).__init__()
        self.model_name = config.LEONARD_NET_NAME
        
        if nb_hidden_layers < 1 :
            raise Exception("Minimum 1 hidden layer")
        
        self.image_net = image_net
        
        self.hiddens = nn.ModuleList()
        
        if nb_hidden_layers > 0:
            self.hiddens = nn.ModuleList([nn.Linear(hidden_layer, hidden_layer) for i in range(nb_hidden_layers-1)])

            self.hiddens.insert(0,nn.Linear(config.NUMBER_OF_CLASSES*2, hidden_layer))

            self.output = nn.Linear(hidden_layer, 1)
            
        if nb_hidden_layers == 0:
            self.output = nn.Linear(config.NUMBER_OF_CLASSES*2, 1)

    def forward(self, x):
        #SPLIT x which is of size [N, 2, 14, 14] to two distinct tensors of size [N, 1, 14, 14]
        input1 = x[:,0:1,:,:]   #(batch_size,1,14,14)
        input2 = x[:,1:2,:,:]   #(batch_size,1,14,14)
        
        lefted, lefted_no = self.image_net(input1)
        
        righted, righted_no = self.image_net(input2)
        
        #CONCAT lefted and righted which are of size [N,10] each to a single tensor of size [N,20]
        hid = torch.cat((lefted, righted),1)
        
        for layer in self.hiddens:
            hid = layer(hid)
            hid = F.relu(hid)
        
        out = self.output(hid)
        
        return torch.sigmoid(out), lefted_no, righted_no