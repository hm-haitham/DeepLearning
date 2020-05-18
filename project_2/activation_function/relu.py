from module import Module 

class ReLU(Module):

    def __init__(self):

        super(ReLU, self).__init__()
        self.x = None

        
    def forward(self,x):

        self.x = x
        y = x * (x > 0).float()
        return y

    def backward(self, dl_ds):
        
        return 2* (self.x.sign() + 1 ) * dl_ds