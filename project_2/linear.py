'''
Fully connected layer 
Parameters : dim_in and dim_out. 
Weights initialization : xavier_normal initilaization and the bias normally initialized.
'''
from xavier_init import Xavier_init
from module import Module
import torch

class Linear(Module):
    
    def __init__(self, dim_in, dim_out,w=None,b=None):
        
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        #input
        self.x = None
        
        # Initialize weights with Xavier initalization or with loaded weights
        if w is None:
            self.w = Xavier_init(self.dim_out, self.dim_in).initialize()
        else:
            self.w = w
            
        if b is None:
            self.b = Xavier_init(1, self.dim_out).initialize()
        else:
            self.b = b
            
        # Initialize gradient
        self.dl_dw = torch.empty(self.w.size())
        self.dl_db = torch.empty(self.b.size())
    
    def forward(self, x):
        """ 
        Computes  a forward pass 
        Input : 
            x : (N, dim_in), with N the number of samples (the batch size)
        Output:
            y : y=xW'+ b with W:(dim_out, dim_in), b:(N, dim_out) giving y:(N, dim_out)
        """
        self.x = x
        return self.x.mm(self.w.t()) + self.b

    def backward(self, dl_ds):
        """ 
        Computes  a backward pass and update gradient of the layer's parameters
        Input : 
            dl_ds : gradient with repect to the activation
        Output:
            dl_dx : gradient with repect to the input
        """
        
        ds_dx = self.w.t()
        dl_dx = ds_dx.mm(dl_ds)
        
        self.dl_db = grad_wrt_output
        self.dl_dw = self.x.mm(dl_ds)
  
        return dl_dx
        
    def param(self):
        return [[self.w , self.dl_dw],[self.b, self.dl_db]]

     
    def reset_gradient(self):
        
        self.dl_dw.zero_()
        self.dl_db.zero_()

    def reset_params(self):
 
        self.w = Xavier_init(self.dim_out, self.dim_in).initialize()
        self.b = Xavier_init(self.dim_out, 1).initialize()