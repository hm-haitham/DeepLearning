class MomentumSGDOptimizer(Optimizer):
    
    def __init__(self, model, lr, momentum):
    
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.prev_grad = [torch.zeros(p[0].shape) for p in self.model.param()]
        
    def step(self):
        
        index = 0
        for m in self.model.modules : 
            if not m.param():  #True if param not empty
                for i, p in enumerate(m.param()): #[weight and bias] 
                    
                    #weights
                    if(i == 0):
                        #v_t = momentum * v_t-1 + (1 - momentum) * grad_t
                        self.prev_grad[index] = self.momentum * self.prev_grad[index] + (1- self.momentum) * p[1]
                        m.w = m.w - self.prev_grad[index]
                        index += 1
                    
                    #bias    
                    elif(i == 1): 
                        #v_t = momentum * v_t-1 + (1 - momentum) * grad_t
                        self.prev_grad[index] = self.momentum * self.prev_grad[index] + (1- self.momentum) * p[1]
                        m.b = m.b - self.prev_grad[index]
                        index += 1
                        
                    else:
                        raise Exception('Parameters unknown')
            