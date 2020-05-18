from module import Module
import torch 
class CrossEntropyLoss(Module):
    
    def __init__(self,model):
        
        super(CrossEntropyLoss, self).__init__()
        self.x = None
        self.y = None
        self.model = model
        
    def softmax(self,x):
        # to make exp stable
        x_shift = torch.max(x,1,keepdim=True)[0]
        
        x_exp = torch.exp(x-x_shift)
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        
        return x_exp/x_exp_sum
        
    def forward(self, x, y):
        '''
        Forward pass 
            x: (batchsize, dim)
            target: (batchize, dim)
            return: CrossEntropyLoss (1)
        '''
        self.x = x
        self.y = y
        m = y.shape[0] 
        s = self.softmax(x)
        
        log_likelihood = -y * torch.log(s)
        loss = log_likelihood.sum(1)
        return loss.mean(0)
    
    def backward(self):
        
        batchsize = self.y.shape[0]
        s = self.softmax(self.x)
        dloss = (self.y-s)/batchsize
        
        #proagate the loss to the model
        self.model.backward(dloss)