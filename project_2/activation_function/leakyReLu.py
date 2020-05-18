from module import Module 

class LeakyReLU(Module):

    def __init__(self,negative_slope=0.01):

        super(LeakyReLU, self).__init__()
        self.x = None
        self.negative_slope = negative_slope

    def forward(self,x):
        
        self.x = x
        y = x * self.negative_slope * (x <= 0).float() +  x * (x > 0).float()
        return y

    def backward(self, dl_ds):
        
        ds_dx =  self.negative_slope * (self.x <= 0).float() +  (self.x > 0).float()
        dl_dx = ds_dx * dl_ds
        return dl_dx
