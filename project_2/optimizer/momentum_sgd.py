from optimizer.optimizer import Optimizer
import torch 

class MomentumSGDOptimizer(Optimizer):
    
    def __init__(self, model, lr = 0.01, momentum = 0.9):
    
        self.model = model
        self.lr = lr
        self.momentum = momentum
        #first moment estimator 
        self.prev_grad = [torch.zeros(p[0].shape) for p in self.model.param()]
        
    def step(self):
        
        index = 0
        for m in self.model.modules : 
            if m.param():  #True if param not empty     
                for i, p in enumerate(m.param()): #[weight and bias] 
                    
                    #compute estimator
                    #v_t = momentum * v_t-1 + lr * grad_t
                    self.prev_grad[index] = self.momentum * self.prev_grad[index] + self.lr * p[1]

                    #weights
                    if(i == 0):
                        m.w = m.w - self.prev_grad[index]
                        index += 1
                    
                    #bias    
                    elif(i == 1): 
                        m.b = m.b - self.prev_grad[index]
                        index += 1
                        
                    else:
                        raise Exception('Parameters unknown')
            