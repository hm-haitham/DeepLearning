import math
import torch 

def build_data(n):
       
    x = torch.empty(n, 2) #(n,2)
    x = x.uniform_(0, 1)
    
    x_centered = x - 0.5
    norm_squared = x_centered.pow(2).sum(dim=1)  #(n,)
    
    r_squared = 1/ (2*math.pi)
    
    y = norm_squared.sub(r_squared).sign().add(1).div(2)
    return x, y

def one_hot_encoding(target):
    
    encoded = torch.empty(target.shape[0], 2)
    for i in range(target.shape[0]): 
        if(target[i]):
            encoded[i,0] = 0
            encoded[i,1] = target[i]
        else : 
            encoded[i,0] = 1
            encoded[i,1] = target[i]
    return encoded
