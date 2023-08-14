from typing import List, Optional
import numpy as np


from tensorgrad.engine import Tensor


class Optimizer(object):
    def __init__(
            self,
            parameters: List[Tensor],
            lr: float,
            momentum: Optional[float] = 0.9
    ):
        self.momentum = momentum
        self.lr = lr
        self.parameters = parameters
        self._running_mean = [np.empty_like(p.grad) for p in self.parameters()]
        self._num_steps = 0

    def zero_grad(self):
        for param in self.parameters():
            param.grad.fill(0)

    def step(self):
        """Update the weights based on the update rule for this optimizer."""
        self._update_running_mean()
        self._num_steps += 1

    def _update_running_mean(self):
        for i, p in enumerate(self.parameters()):
            if self._num_steps == 0:
                self._running_mean[i] = p.grad.copy()
            elif self.momentum is None:
                # TODO: real running mean using num_steps
                raise NotImplementedError
            else:
                self._running_mean[i] = (1 - self.momentum) * (p.grad + self.momentum * self._running_mean[i])
            

class SGD(Optimizer):
    def step(self):
        super(SGD, self).step()
        for i, param in enumerate(self.parameters()):
            #param.data -= self.lr * self._running_mean[i]
            #using simple grad subtraction while testing
            before = param.data.copy()
            print(param.grad)
            param.data = param.data - (self.lr*param.grad)
