from module import Module

class MSELoss(Module):
    
    def __init__(self,model):
        
        super(MSELoss, self).__init()
        self.prediction = None
        self.target = None
        self.model=model
        
    def forward(self, prediction, target):
        
        self.prediction = prediction
        self.target = target
        batched_error = (prediction - target).pow(2).sum(1)
        return batched_error.mean(0)
    
     def backward(self):
        
        batchsize = self.prediction.shape[0]
        dloss = 2*(self.prediction-self.target)/batchsize
        self.model.backward(dloss)