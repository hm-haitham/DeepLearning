from module import Module

class LossMSE(Module):
    
    def __init__(self,model):
        
        super(LossMSE, self).__init__()
        self.prediction = None
        self.target = None
        self.model = model
        
    def forward(self, prediction, target):
        '''
        Forward pass 
            prediction: (batchsize, dim)
            target: (1, dim)
            return: MSE (1)
        '''
        self.prediction = prediction
        self.target = target
        batched_error = (prediction - target).pow(2).sum(1) #L2 norm of the difference between prediction and target
        return batched_error.mean(0)  #take the mean over the batch
    
    def backward(self):
        
        if(self.prediction == None):
            raise Exception("Run backward twice !")
            
        batchsize = self.prediction.shape[0]
        dloss = 2*(self.prediction - self.target)/batchsize
        
        #reinitiliaze the predictions
        self.prediction = None
        
        #proagate the loss to the model
        self.model.backward(dloss)