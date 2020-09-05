import numpy as np

class Tensor():
    def __init__(self,data,_children=(),op=''):
        self.data = np.array(data)
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self.op = op 
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

    def sum(self):
        sum = 0
        for i in self.data:
            sum += i
        out = Tensor(sum,_children=(self,),op='sum')
        def _backward():
            self.grad = np.zeros((*self.data.shape,))
            self.grad += out.grad

        out._backward = _backward
        return out
    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,_children=(self,other),op='+')
        

        def _backward():
            self.grad = np.zeros(self.data.shape)
            other.grad = np.zeros(other.data.shape)
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        
        return out
    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data*other.data,_children=(self,other),op='+')
        

        def _backward():
            print(other.data*out.grad)
            self.grad = np.zeros((*self.data.shape,))
            other.grad = np.zeros((*other.data.shape,))
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad

        out._backward = _backward

        return out
    def __pow__(self,other):
        out = Tensor(self.data**other,_children=(self,other),op='pow')

        def _backward():
            self.grad = np.zeros((*self.data.shape,))
            self.grad +=(other*self.data**other-1)*out.grad
            
    def __neg__(self): # -self
        return self*-1

    def __radd__(self, other): # other + self
        return self+other

    def __sub__(self, other): # self - other
        return self+(-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self*other

    def __truediv__(self, other): # self / other
        return self*other**-1

    def __rtruediv__(self, other): # other / self
        return other*self**-1
    def __repr__(self):
        return f"Tensor with values:{self.data} and Gradient of:{self.grad}"
