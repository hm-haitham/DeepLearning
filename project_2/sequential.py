from module import Module 

class Sequential(Module):
    
    def __init__(self, *modules):
        
        super(Sequential, self).__init__()
        self.modules = []
        
        for m in modules:
            self.modules.append(m)
    
    def forward(self, x):
        
        for m in self.modules:
            x = m.forward(x)
        return x
        
    def backward(self, grad):
        
        for m in self.modules[::-1]:
            grad = m.backward(grad)
        return grad
    
    def param(self):
        
        parameters = []
        for m in self.modules:
            parameters += m.param
        return parameters