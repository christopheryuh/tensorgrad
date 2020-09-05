import torch
import numpy as np
from tensorgrad.engine import Tensor

xpt = torch.Tensor([2,3])
xpt.requires_grad = True
ypt = torch.Tensor([2,2])
zpt = xpt*ypt
wpt = sum(zpt)
wpt.backward()

x = Tensor([1,2])
y = Tensor([2,2])

z = y*x
print(z)
w = z.sum()
print(w,wpt)
w.backward()
print(x,xpt,xpt.grad)



