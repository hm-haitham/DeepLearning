'''
Fully connected layer 

Parameters : dim_in and dim_out. 
Weights initialization : xavier_normal initilaization and the bias normally initialized.
'''
import Xavier_init
import torch

class Linear(Module):
    
    def __init__(self, dim_in, dim_out):
        
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        #input
        self.x = 0
        
        # Initialize weights with Xavier initalization
        self.w = Xavier_init(self.dim_out, self.dim_in).initialize()
        self.b = Xavier_init(self.dim_out, 1).initialize()

        # Initialize gradient
        self.dl_dw = torch.empty(self.w.size())
        self.dl_db = torch.empty(self.b.size())
    
    def forward(self, x):
        
        self.x = x
        return self.w.mm(self.x) + self.b

    def backward(self, dl_ds):
       
        ds_dx = self.w.t()
        dl_dx = ds_dx.mm(dl_ds)
        
        self.dl_db = grad_wrt_output
        self.dl_dw = self.x.mm(dl_ds)
  
        return dl_dx
        
    def params(self):
        
        return [self.w, self.bias]
    
    def update(self, lr):
        
        self.w = self.w - lr * self.dl_dw
        self.b = self.b - lr * self.dl_db
        
    def reset_gradient(self):
        
        self.dl_dw.zero_()
        self.dl_db.zero_()

    def reset_params(self):
 
        self.w = Xavier_init(self.dim_out, self.dim_in).initialize()
        self.b = Xavier_init(self.dim_out, 1).initialize()
