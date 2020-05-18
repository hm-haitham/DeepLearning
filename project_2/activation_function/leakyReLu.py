from module import Module 

class LeakyReLU(Module):

    def __init__(self,negative_slope=0.01):

        super(LeakyReLU, self).__init__()
        self.x = None
        self.negative_slope=negative_slope


    def forward(self,x):

        self.x = x
        if (x>0):
            return x
        else:
            return x * negative_slope


    def backward(self, dl_ds):

        if (self.x>0):
            return dl_ds
        else:
            return dl_ds * negative_slope

    def param(self) :
        return []