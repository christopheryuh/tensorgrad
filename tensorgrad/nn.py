import numpy as np 
from tensorgrad.engine import random,Tensor

class Module():
    def zero_grad(self):
        for p in self.parameters():
            p.grad[:] = 0



class Conv2d(Module):
    def __init__(self,):
        pass
    def __call__(self):
        pass


class Linear(Module):
    def __init__(self,n_in,n_out,use_bias = True):
        self.use_bias = use_bias
        self.w = random(size=(n_out,n_in,))
        self.b = random(size=(n_out,1))
    def parameters(self):
        return [self.w,self.b]
    def __call__(self,x):
        print(self.w.shape,x.shape)
        x = x if isinstance(x,(Tensor)) else Tensor(x)
        x = self.w @ x
        if self.use_bias:
            x = x + self.b
        return x

