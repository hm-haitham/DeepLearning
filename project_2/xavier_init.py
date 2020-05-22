import math
import torch 
from initializer import Initializer

class Xavier_init(Initializer):
    
    def __init__(self, fan_in, fan_out, dist = 'uniform'):
        
        super(Xavier_init, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.dist = dist 
    
    def initialize(self):
        
        if(self.dist == 'uniform'):
            a =  math.sqrt(6. / (self.fan_in + self.fan_out) )    
            w = torch.empty(self.fan_out, self.fan_in).uniform_(-a,a)

        #using normal distribution
        else: 
            std = math.sqrt(2. / (self.fan_in + self.fan_out) )
            w = tensor.normal_(0,std)
        
        return w
            
            
        
        
         