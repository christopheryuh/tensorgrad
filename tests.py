import torch
import numpy as np
from tensorgrad.engine import Tensor

#TODO: make this a real testing script
xpt = torch.Tensor([1,2])
xpt.requires_grad = True
ypt = torch.Tensor([2,2])
zpt = xpt*ypt
wpt = sum(zpt)
wpt.backward()


x = Tensor([1,2])
y = Tensor([2,2])
z = x * y
w = sum(z)
w.backward()

