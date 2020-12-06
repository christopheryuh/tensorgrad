from tensorgrad import engine
import numpy as np
import math

def random(*args,**kwargs):
    return engine.Tensor(np.random.uniform(*args,**kwargs))

def glorot_uniform(fan_in, fan_out, **kwargs):

    limit = math.sqrt(6 / (fan_in + fan_out))

    return engine.Tensor(np.random.uniform(-limit, limit, **kwargs))

def empty(*args,**kwargs):
    return engine.Tensor(np.empty(*args,**kwargs))

def zeros(*args, **kwargs):
    return engine.Tensor(np.zeros(*args,**kwargs))





def concat(tensors,axis=0):
    """Concatenate the tensors along the given axis"""
    tensors = [t if isinstance(t, engine.Tensor) else engine.Tensor(t) for t in tensors]
    out = engine.Tensor(np.concatenate([t.data for t in tensors], axis=axis),_children=tensors, op='concatenate')

    def _backward():
        st = 0
        for t in enumerate(tensors):
            indexing = [slice(st, st + d) if i == axis else slice(0,d) for i, d in enumerate(t.shape)]
            t.grad += out.grad[indexing]
            st += t.shape[axis]
    out._backward = _backward
    return  out



def oneHot(x,depth=None):
    x = np.array(x)

    assert (depth != None)
    zeros = np.zeros((x.shape[0],depth))
    x = x.reshape((-1,))
    for idx,val in enumerate(x):
        zeros[idx][val] = 1

    return zeros


def assign(x, indexing, y):
    y = y if isinstance(y, engine.Tensor) else engine.Tensor(y)
    data = x.data.copy()
    data[indexing] = y.data
    out = engine.Tensor(data, _children=(x, y), op='set item',)

    def _backward():
        y.grad += out.grad[indexing]
        gradient = out.grad.copy()
        x.grad += gradient
        x.grad[indexing] = 0.

    out._backward = _backward

    return out
    