import torch
import numpy as np
from tensorgrad.engine import Tensor

xpt = torch.Tensor([1,2])
xpt.requires_grad = True
ypt = torch.Tensor([2,2])
ypt.requires_grad = True
zpt =  xpt*ypt
wpt = sum(zpt)

wpt.backward()

print(xpt.grad)
print(wpt)

x = Tensor([1,2])
y = Tensor([2,2])

z = y*x
w = z.sum()
print(w,wpt)
w.backward()
print(x.grad)
assert x.grad == xpt.grad


