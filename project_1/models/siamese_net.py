import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import config

class siamese_net(nn.Module):

    def __init__(self, weight_sharing = True, architecture = 1, nb_channels = 24, nb_hidden = 300):
        
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
            self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(1, nb_channels , kernel_size=3),  
                                   nn.LeakyReLU(),
                                   nn.Dropout(p=0.2),                  
                                   nn.Conv2d(nb_channels, 2 * nb_channels, kernel_size=3),
                                   nn.LeakyReLU(),           
                                   nn.MaxPool2d(2),
                                   nn.Dropout(p=0.2)) for i in range(self.nb_subnet) ] )
        
            #fully connected part   
            # from a conv (5,5)  => fc layer (25,512) => Relu => fc layer (512,10) => Softmax
            
            out_conv = 25 * 2 * nb_channels                                          
            
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(out_conv , nb_hidden), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                                   nn.Linear(nb_hidden, 10) ) for i in range(self.nb_subnet) ] )
        if (architecture == 2 ):
            
            self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(1, nb_channels, kernel_size=5, stride=1, padding=2),
                                                     nn.ReLU(),
                                                     nn.MaxPool2d(kernel_size=3, stride=3),
                                                     nn.Dropout(p=0.2),
                                                     nn.Conv2d(nb_channels, 2 * nb_channels, kernel_size=5, stride=1, padding=2),
                                                     nn.ReLU(),
                                                     nn.MaxPool2d(kernel_size=3, stride=3),
                                                     nn.Dropout(p=0.2)) for i in range(self.nb_subnet) ] )
            out_conv = 1 * 2 * nb_channels                                          
            
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(out_conv , nb_hidden), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                                   nn.Linear(nb_hidden, 10) ) for i in range(self.nb_subnet) ] )

        if (architecture == 3 ):
            
            #(W−F+2P)/S+1
           
            self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(1, nb_channels, kernel_size=3, stride=1, padding=0),  #(14-3-0) = 12 
                                                     nn.LeakyReLU(),
                                                     nn.MaxPool2d(kernel_size=3, stride=3), #4
                                                     nn.Dropout(p=0.2),
                                                     nn.Conv2d(nb_channels, 2 * nb_channels, kernel_size=2, stride=1, padding=0), #2
                                                     nn.LeakyReLU(),
                                                     nn.MaxPool2d(kernel_size=2, stride=2), #1
                                                     nn.Dropout(p=0.2)) for i in range(self.nb_subnet) ] )
                                                     
            out_conv = 1 * 2 * nb_channels                                          
            
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(out_conv , nb_hidden), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                                   nn.Linear(nb_hidden, 10) ) for i in range(self.nb_subnet) ] )
        
        if (architecture == 4 ):
            
            #(W−F+2P)/S+1
            self.conv = nn.ModuleList([nn.Sequential(nn.Conv2d(1, nb_channels, kernel_size=2, stride=1, padding=1),  #(14-2+2)+1 = 15
                                                     nn.LeakyReLU(),
                                                     nn.MaxPool2d(kernel_size=3, stride=3), #5
                                                     nn.Dropout(p=0.2),
                                                     nn.Conv2d(nb_channels, 2 * nb_channels, 
                                                               kernel_size=3, stride=1, padding=0), #(5-3)+1 = 3
                                                     nn.LeakyReLU(),
                                                     nn.MaxPool2d(kernel_size=3, stride=3), #1  
                                                     nn.Dropout(p=0.2) ) for i in range(self.nb_subnet) ] )
            
            out_conv = 1 * 2 * nb_channels 
            
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(out_conv , nb_hidden), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                                   nn.Linear(nb_hidden, 10) ) for i in range(self.nb_subnet) ] )
            
        #combine the subnetwork output
        #(20,1) and sigmoid
        self.comb = nn.Sequential(nn.Linear(20, 300), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                  nn.Linear(300, 300), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                  nn.Linear(300, 1), nn.Sigmoid() ) 
     
            
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