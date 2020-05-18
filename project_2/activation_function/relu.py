from module import Module 

class ReLU(Module):

    def __init__(self):

        super(ReLU, self).__init__()
        self.x = None

        
    def forward(self,x):

        self.x = x
        if (x>0):
            return x
        else:
            return 0


    def backward(self, dl_ds):

        if (self.x>0):
            return dl_ds
        else:
            return 0