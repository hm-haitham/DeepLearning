class Tanh(module):
    
    def __init__(self):
        
        super(Tanh, self).__int__()
        x = 0
        
    def forward(self, x):
        
        self.x = x
        return x.tanh()
    
    def backward(self, dl_ds):
        
        ds_dx = 1 - x.tanh().pow(2)
        dl_dx = ds_dx * dl_ds
        return ds_dx * dl_ds
    
    def params(self):
        return []