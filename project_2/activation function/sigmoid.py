class Sigmoid(Module):
    
    def __init__(self):
        
        super(Sigmoid, self).__init__()
        self.x = 0
    
    def sigmoid(self,x):
        
        s = 1 / (1 + x.mul(-1).exp() ) 
        return s
    
    def forward(self,x):
        
        self.x = x
        return self.sigmoid(x)

    def backward(self, dl_ds):
        
        sigm = self.sigmoid(self.x)
        ds_dx = sigm * (1 - sigm)
        dl_dx = ds_dx * dl_ds
        return ds_dx * dl_ds
    
    def param(self) :
        return []
