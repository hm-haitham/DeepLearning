import math
import torch 

class Xavier_init(Initalizer):
    
    def __init__(self,dim_in, dim_out, weights, dist = 'uniform'):
        super(Xavier_init, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dist = dist 
    
    def initialize(self):
        
        if(self.dist == 'uniform'):
            a =  math.sqrt(6. / (fan_in + fan_out) )    
            w = torch.empty(fan_out, fan_in).uniform_(-a,a)

        #normal
        else: 
            std = math.sqrt(2. / (fan_in + fan_out) )
            w = tensor.normal_(0,std)
        
        return w
            
            
        
        
         