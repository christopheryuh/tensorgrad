import torch
import numpy as np
from tensorgrad.engine import Tensor

#TODO: make this a real testing script

def helper(fn,fn_input):
    xpt = torch.Tensor(fn_input)
    xpt.requires_grad = True

    ypt = fn(xpt,pytorch=True)
    ypt.backward()

    x = Tensor(fn_input)
    
    y = fn(x)

    y.backward()

    assert np.all(y.data == ypt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())

def test_same_shape_add():
    fn_input = np.array([1,2,3])
    def fn(x,pytorch=False):
        if pytorch:      
            z = x + torch.tensor([4,3,2])
            return z.sum()
        else:
            z = x + np.array([4,3,2])
            return z.sum()
    helper(fn,fn_input)

def test_mul():
    fn_input = np.array([[1,2],[1,2]])
    def fn(x,pytorch=False):
        if pytorch:
            z = x * torch.tensor([[1,2],[1,2]])
            h = z * torch.tensor([1,2])
            return h.sum()
        else:
            z = x * np.array([[1,2],[1,2]])
            h = z * np.array([1,2])
            return h.sum()
    helper(fn,fn_input)

test_same_shape_add()
test_mul()
