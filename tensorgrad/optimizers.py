from typing import List, Optional
from tensorgrad.engine import *
import numpy as np

class Optimizer():
    def __init__(self,parameters,lr,momentum:Optional[float] = 0.9):
        self.momentum = momentum
        self.lr =lr
        self.parameters = parameters
        self._running_mean = [np.empty_like(p.grad) for p in self.parameters]
    
    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)
    
    def step(self):
        """Update the weights based on the update rule for this optimizer"""

        if self.momentum is not None:
            self._update_running_mean()
            # Take the updates from the running means
        else:
            #Take update from the gradient itself
            pass

    def _update_running_mean(self):
        for p, rm in zip(self.parameters, self._running_mean):
            rm = (1 - self.momentum) * (p.grad + self.momentum * rm)        

class SGD(Optimizer):
    def step(self):
        super(SGD, self).step()
        for i,param in enumerate(self.parameters):
            param.data = self.lr * (param.data - self._running_mean[i])