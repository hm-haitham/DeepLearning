import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class SiameseNet(nn.Module):

    def __init__(self, left_net, right_net = None, nb_hidden_layers = config.SIAMESE_NET_NB_HIDDEN, hidden_layer=config.SIAMESE_NET_HIDDEN_LAYER, soft = False):
        
        super(SiameseNet, self).__init__()
        
        if nb_hidden_layers < 1:
            raise Exception("Minimum 1 hidden layers for the Siamese Net")
        
        self.subnets = nn.ModuleList([left_net])
        
        self.soft = soft
        
        if right_net == None :
            self.weight_sharing = True
        else :
            self.weight_sharing = False
            self.subnets.append(right_net)
        
        self.hiddens = nn.ModuleList([nn.Sequential(nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)) for i in range(nb_hidden_layers-1)])

        self.hiddens.insert(0,nn.Sequential(nn.Linear(config.NUMBER_OF_CLASSES*2, hidden_layer), nn.LeakyReLU(), nn.Dropout(p=0.2)))

        self.output = nn.Linear(hidden_layer, 1)

        
    def forward(self, x):
        
        #SPLIT x which is of size [N, 2, 14, 14] to two distinct tensors of size [N, 1, 14, 14]
        input1 = x[:,0:1,:,:]   #(batch_size,1,14,14)
        input2 = x[:,1:2,:,:]   #(batch_size,1,14,14)
        
        lefted, lefted_no = self.subnets[0](input1)   #softmax(output), output
        
        if self.weight_sharing :
            righted, righted_no = self.subnets[0](input2)
        else :
            righted, righted_no = self.subnets[1](input2)
        
        #CONCAT lefted and righted which are of size [N,10] each to a single tensor of size [N,20]
        if(self.soft):
            hid = torch.cat((lefted, righted),1)
        else:
            hid = torch.cat((lefted_no, righted_no), 1)
        
        for block in self.hiddens:
            hid = block(hid)
        
        out = self.output(hid)
        
        return torch.sigmoid(out), lefted_no, righted_no