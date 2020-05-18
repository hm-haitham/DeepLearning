from optimizer.optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, model, lr):
        
        super(SGD, self).__init__()
        
        self.model = model
        self.lr = lr
    
    def step(self):
    
        for m in self.model.modules :  
            for i, _ in enumerate(m.param()): #[weight and bias] or []
                if(i == 0):
                    m.w = m.w - self.lr * m.dl_dw
                elif(i == 1): 
                    m.b = m.b - self.lr * m.dl_db
                else:
                    raise Exception('Parameters unknown')
                
                