import numpy as np

from tensorgrad import func
from tensorgrad.func import *


class Tensor():
    def __init__(self, data, _children=(), op=''):
        self.data = np.array(data)
        if len(self.data.shape) == 0:
            self.data = self.data.reshape(1) 
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data).astype(np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self.op = op 

    def __array__(self):
        return self.data

    def backward(self):
        """Perform backpropagation on this Tensor.

        For every Tensor object t in self's computation graph, set t.grad to the partial derivative
        of self with respect to t.

        Recursively calls t._backward() for all such t.

        """

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for t in reversed(topo):
            t._backward()
    


    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape))

        def _backward():
            self.grad = self.grad.reshape(self.data.shape) * out.grad         
        out._backward = _backward

        return out

    def broadcast(self, other):

        if isinstance(other, (float, int)) or (isinstance(other, Tensor) and other.shape == (1,)):
            return self

        other = other if isinstance(other, (Tensor)) else Tensor(other)

        for ax in range(len(self.data.shape)):
            xd = self.data.shape[ax]
            yd = other.data.shape[ax]
            if xd == yd:
                pass

            if yd == 1 and xd > 1:
                pass

            elif xd == 1 and yd > 1:
                out = func.concat([self.data for _ in range(yd)], axis=ax)

            else:
                out = self
                #because both sides need to use it dont throw error
                #raise ValueError(f'Mismatched Shape, {self.shape} and {other.shape}')

        return out

    def relu(self):
        out = Tensor(np.where(self.data > 0, self.data, 0))

        def _backward():
            self.grad += np.where(self.data < 0, self.grad, self.data) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(self.data)))

        def _backward():
            self.grad += ((1 - out.data) * out.data) * out.grad
        out._backward = _backward

        return out

    def softmax(self):
        x = self.data - np.max(self.data)           # shift for numerical stability
        ex = np.exp(x)
        out =  Tensor(ex / np.sum(ex,axis=-1,keepdims=True))
        def _backward():
            self.grad = out.grad * np.sum((out.data - 1 / np.exp(self.data)),axis=-1, keepdims=True)

        out._backward = _backward
        return out

    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis), _children=(self,), op='sum')

        def _backward():
            if axis == None:
                self.grad += out.grad
            else:
                self.grad += out.grad.reshape(*[d if d != axis else 1 for d in self.data.shape])

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # if self.shape == other.shape:
        #     pass
        # else:
        #     self = self.broadcast(other)
        #     other = other.broadcast(self)
        out = Tensor(self.data + other.data, _children=(self, other), op='+')

        def _backward():
            

            print(self.grad.shape,out.grad.shape)
            self.grad = self.grad + out.grad
            if self.grad.shape == out.grad.shape:
                self.grad = self.grad + out.grad
            
            else:
                axises = ()
                for ax in range(len(self.grad.shape)):
                    if out.grad.shape[ax] > self.grad.shape[ax]:
                        axises = axises + (ax,)

                self.grad = (self.grad + out.grad).sum(axis=axises)


            if other.grad.shape == out.grad.shape:
                other.grad = other.grad + out.grad
            
            else:
                axises = ()
                for ax in range(len(other.grad.shape)):
                    if out.grad.shape[ax] > other.grad.shape[ax]:
                        axises = axises + (ax,)

                other.grad = (other.grad + out.grad).sum(axis=axises)
                #print(other.grad.shape)
            

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # if self.shape == other.shape:
        #     pass
        # else:
        #     self = self.broadcast(other)
        #     other = other.broadcast(self)

        out = Tensor(self.data * other.data, _children=(self, other), op='+')

        def _backward():
            self.grad = np.zeros(self.data.shape)
            other.grad = np.zeros(other.data.shape)
            self.grad = self.grad + (other.data * out.grad)
            other.grad = other.grad + (self.data * out.grad)

        out._backward = _backward

        return out


    def __pow__(self, other):
        other = float(other)
        out = Tensor(self.data**other, _children=(self), op='pow')

        def _backward():
            self.grad = np.zeros(self.data.shape)
            self.grad += (other * self.data**other - 1) * out.grad
        
        out._backward = _backward
        return out 

    def __getitem__(self, other):
        out = Tensor(self.data[other], _children=(self,), op='get item')

        def _backward():
            self.grad[other] += out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(np.matmul(self.data, other.data), _children=(self, other), op='matmul')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += (out.grad.T @ self.data).T

        out._backward = _backward
        return out


    def __setitem__(self, idx, other):
        raise ValueError

    def __neg__(self):  # -self
        return self * (np.ones(self.shape)*-1)

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1.0

    def __rtruediv__(self, other):  # other / self
        return other / self

    def __repr__(self):
        return f"Tensor with values:{self.data} and Gradient of:{self.grad} and shape of:{self.shape}"
