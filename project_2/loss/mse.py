class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init()
        
    def forward(self, s, y):
    
        batched_error = (s - y).pow(2).sum(1)
        return batched_error.mean(0)
    
    def param(self) :
        return []
