from optimizer.optimizer import Optimizer
import torch 

class RMSPropOptimizer(Optimizer):
    def __init__(self, model, lr = 0.001, gamma = 0.9, epsilon=1e-8):
        
        self.model = model
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        #second moment estimator for each parameter
        self.prev_grad = [torch.zeros(p[0].shape) for p in self.model.param()]


    def step(self):
        
        index = 0
        for m in self.model.modules : 
            if m.param():  #True if param not empty     
                for i, p in enumerate(m.param()): #[weight and bias] 
                    
                    #compute estimator 
                    #v_t = gamma * v_t-1 + (1 - gamma) * grad_t**2
                    self.prev_grad[index] = self.gamma * self.prev_grad[index] + (1 - self.gamma) * p[1]**2
                    
                    #weights
                    if(i == 0):            
                        m.w = m.w - self.lr / (self.prev_grad[index]+ self.epsilon).sqrt() * p[1]
                        index += 1
                    
                    #bias    
                    elif(i == 1):                         
                        m.b = m.b - self.lr / (self.prev_grad[index] + self.epsilon).sqrt() * p[1]
                        index += 1
                        
                    else:
                        raise Exception('Parameters unknown')
            
            