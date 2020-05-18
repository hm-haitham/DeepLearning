from module import Module 

class Tanh(Module):
    
    def __init__(self):
        
        super(Tanh, self).__init__()
        x = None
        
    def forward(self, x):
        
        self.x = x
        return x.tanh()
    
    def backward(self, dl_ds):
        
        ds_dx = 1 - self.x.tanh().pow(2)
        dl_dx = ds_dx * dl_ds
        return ds_dx * dl_ds
    
    def params(self):
        return []