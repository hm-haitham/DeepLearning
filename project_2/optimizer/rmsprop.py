class RMSPropOptimizer(object):
    def __init__(self, model, lr = 0.001, gamma = 0.9, eps=1e-8):
        
        self.model = model
  
        self.lr = lr
    
        self.beta = beta
        self.r = None
        self.eps = eps

    def step(self):
        if self.r is None:
            self.r = [torch.zeros(param[0].shape) for param in self.params]

        for i, param in enumerate(self.params):
            self.r[i] = self.beta*self.r[i]  + (1 - self.beta) * param[1] * param[1]
            param[0] -= self.eta * param[1]/(torch.sqrt(self.r[i]) + self.eps)