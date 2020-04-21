import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class siamese_net(nn.Module):

    def __init__(self, weight_sharing = True, architecture = 1 ):
        
        super(siamese_net, self).__init__()
        
        #if weight sharing forward on the same subnet
        if weight_sharing: 
            self.nb_subnet = 1
        else:
            self.nb_subnet = 2
            
        if (architecture == 1 ):
            # subnetwork 1
            # (1,14,14) conv (1,3,3) =>  (1,12,12)
            # Relu 
            # (1,12,12) conv (1,3,3) => (1,10,10)
            # Relu
            # (1,10,10) maxpool (2,2) => (5,5)
            # try to add different channels and add dropout ? 
            self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(1, 1 , kernel_size=3),  
                                   nn.LeakyReLU(),
                                   nn.Conv2d(1, 1 , kernel_size=3),
                                   nn.LeakyReLU(),
                                   nn.MaxPool2d(2) ) for i in range(self.nb_subnet) ] )
        
            #fully connected part   
            # from a conv (5,5)  => fc layer (25,512) => Relu => fc layer (512,10) => Softmax
            
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(25, 512),
                                     nn.LeakyReLU(),
                                     nn.Linear(512, 10),
                                     nn.Softmax() ) for i in range(self.nb_subnet) ] )
        #combine the subnetwork output
        #(20,1) and sigmoid
        self.comb = nn.Sequential(nn.Linear(20, 1), nn.Sigmoid())
        
    def forward_once(self, x, subnet_nbr):
        
        x = self.conv[subnet_nbr](x)  #(batch_size,1,5,5)
        
        #need to flatten to insert in the fc layer
        x = x.view(x.size(0), -1)    #(batch_size,25)
        
        x = self.fc[subnet_nbr](x)   #(batch_size,10)
        
        return x
    
    def forward(self, input1, input2): 
        
        #we compute the output for each subnet 
        #apply conv and fc and get 10 units per subnetwork
        output1 = self.forward_once(input1,0)
        output2 = self.forward_once(input2, self.nb_subnet - 1)  #-1 because we access a list index starting at 0 
        
        #concat the two output 
        concat_subnet = torch.cat((output1, output2),1)     #get (batch_size,20)
        
        #forward on combination layer
        output = self.comb(concat_subnet)
        
        return output1, output2, output