from module import Module

class LossBCE(Module):
    
    def __init__(self,model):
        
        super(LossBCE, self).__init__()
        self.prediction = None
        self.target = None
        self.model = model
        self.reg=1e-12 #regularization to avoid log(0) and divide by 0 problems
        
    def forward(self, prediction, target):
        '''
        Forward pass 
            prediction: (batchsize, dim)
            target: (1, dim)
            return: BCE (1)
        '''
        
        self.prediction = prediction
        self.target = target
        batched_error = -(target*prediction.clamp(min=self.reg,max=1).log())-(1-target)*(1-prediction).clamp(min=self.reg,max=1).log()
        return batched_error.mean()  #take the mean over the batch
    
    def backward(self):
        
        if(self.prediction == None):
            raise Exception("Run backward twice !")
            
        batchsize = self.prediction.shape[0]
        
        batched_dloss = (self.prediction - self.target)/((self.prediction - self.prediction*self.prediction).clamp(min=self.reg))
        dloss=batched_dloss/batchsize
        
        #reinitiliaze the predictions
        self.prediction = None
        
        #proagate the loss to the model
        self.model.backward(dloss)