from optimizer.optimizer import Optimizer
import torch 

class AdamOptimizer(Optimizer):
    
    def __init__(self, model, lr = 0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = [torch.zeros(p[0].shape) for p in self.model.param()]
        self.second_moment = [torch.zeros(p[0].shape) for p in self.model.param()]

    def step(self):
        index = 0
        for m in self.model.modules : 
            if m.param():  #True if param not empty     
                for i, p in enumerate(m.param()): #[weight and bias] 

                    #m_t = beta_1 * m_t-1 + (1 - beta_1) * grad_t
                    #v_t = beta_2 * v_t-1 + (1 - beta_2) * grad_t
                    self.first_moment[index] = self.beta1 * self.first_moment[index] + (1 - self.beta1) * p[1]
                    self.second_moment[index] = self.beta2 * self.second_moment[index] + (1 - self.beta2) * p[1]**2
                        
                    #biases corrected 
                    unbias_m = self.first_moment[index] / (1-self.beta1)
                    unbias_v = self.second_moment[index] / (1-self.beta2)
                    
                    #weights
                    if(i == 0):
                        m.w = m.w - self.lr * unbias_m / (unbias_v.sqrt() + self.epsilon)
                        index += 1
                    
                    #bias    
                    elif(i == 1): 
                        m.b = m.b - self.lr * unbias_m / (unbias_v.sqrt() + self.epsilon)
                        index += 1
                    else:
                        raise Exception('Parameters unknown')