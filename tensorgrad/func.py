from tensorgrad import engine
import numpy as np

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