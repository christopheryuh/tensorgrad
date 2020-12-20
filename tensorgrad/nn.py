import numpy as np 

import math

from tensorgrad.engine import Tensor
from tensorgrad import utils
from tensorgrad.optimizers import SGD
from tensorgrad.func import glorot_uniform, assign

from abc import ABC, abstractmethod

epsilon = .001



class Layer():
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def __call__():
        pass

class WeightedLayer(Layer):
    @abstractmethod
    def parameters():
        pass
        


class Activation(Layer):
    def __init__(self):
        self.has_vars = False

    def __call__():
        pass
    
    def parameters():
        return []


class Relu(Activation):
    def __call__(self, x, training=False):
        return x.relu()


class Sigmoid(Activation):
    def __call__(self, x, training=False):
        return x.sigmoid()


class Softmax(Activation):
    def __call__(self, x, training=False):
        return x.softmax()


class Crossentropy():
    def __call__(self, y_hat, y):
        labels = y
        out = Tensor(-1 * labels * np.log(y_hat.data + epsilon))
        out = out.sum()

        def _backward():
            y_hat.grad += -1 * labels * (y_hat.data + epsilon)

        out._backward = _backward
        return out


class Conv2d(WeightedLayer):
    def __init__(self, inputs, outputs, kernel_dim, stride=(1,1), use_bias=True, dilation=(1,1),padding='valid'):

        
        assert padding in {'valid', 'same'}

        self.dilation = dilation
        self.stride = stride

        self.w = glorot_uniform(inputs, outputs, size=(outputs, inputs, kernel_dim, kernel_dim))

        print(np.max(self.w.data.reshape((-1,))))


        assert not np.any(np.isnan(self.w.data)) or not np.any(np.isinf(self.w.data))
    
        self.outs = outputs
        self.use_bias = use_bias 
        self.padding_style = padding



        self.kh = kernel_dim
        self.kw = kernel_dim

        self.kernel_dim = (kernel_dim,kernel_dim)

        self.has_vars = True

        self.h2 = self.kh // 2
        self.w2 = self.kh // 2

        if self.padding_style == 'valid':
            self.padding_dims = [0,0]
        elif self.padding_style == 'same':
            self.padding_dims = [self.h2,self.w2]

        if self.use_bias:
            self.b = zeros((self.outs,))

    def parameters(self):
        if self.use_bias:
            return [self.w, self.b]
        else:
            return [self.w, ]
    

    def get_pixel_value(self,patch, kernel):


        out = patch.reshape((1,-1,self.kh,self.kw)) * kernel

        out = out.sum(axis=3)
        out = out.sum(axis=2)
        out = out.sum(axis=1)

        return out

    def get_output_shape(self,input_shape):
        n, c, h, w = input_shape        
        kh, kw = self.kernel_dim
        h_out = math.floor((h + 2 * self.padding_dims[0] - self.dilation[0] * (kh - 1) - 1) / self.stride[0] + 1)        
        w_out = math.floor((w + 2 * self.padding_dims[1] - self.dilation[1] * (kw - 1) - 1) / self.stride[1] + 1)

        return (n,self.outs, h_out, w_out)


    def _convolve_op(self,x):
        shape = x.shape

        n_out,c_out,h_out,w_out = self.get_output_shape(shape)

        assert self.stride[0] == self.stride[1]

        assert self.h2 == self.w2

        out = np.zeros((*shape[:2],*self.kernel_dim,*shape[2:]))

        for nidx,n in enumerate(x):
            for cidx, c in enumerate(n):

                c = c.reshape(1,1,*x.shape[2:])
                if self.padding_style == 'valid':
                    img = utils.im2col(c,*self.kernel_dim,padding=0,stride=self.stride[0])
                if self.padding_style == 'same':
                    img = utils.im2col(c,*self.kernel_dim,padding=self.h2,stride=self.stride[0])

                print(img.shape)
                
                print(out.shape)

                out[nidx,cidx,:,:,:,:] = img.reshape((*self.kernel_dim,*shape[2:]))

        return Tensor(out)

        def _backward():
            return np.zeros_like(x)

        out._backward = _backward()


    def __call__(self, x, training=False):

        x = np.array(x)

        shape = n,c,h,w = x.shape

        n_out,c_out,h_out,w_out = self.get_output_shape(shape)

        x = self._convolve_op(x)

        x = x.reshape((n,1, c, *self.kernel_dim,h_out, w_out))

        kernel = self.w.reshape((1,self.outs,c,*self.kernel_dim,1,1))

        x = x*kernel

        x = x.sum(axis=2).sum(axis=3).sum(axis=4)

        

class Linear(WeightedLayer):
    def __init__(self, n_in, n_out, use_bias=True):
        self.use_bias = use_bias
        self.has_vars = True
        self.w = glorot_uniform(n_in, n_out, size=(n_in, n_out))
        self.b = zeros((1, n_out))

    def parameters(self):
        if self.use_bias:
            return [self.w, self.b]
        else:
            return [self.w, ]

    def __call__(self, x, training=False):
        x = x if isinstance(x, (Tensor)) else Tensor(x)
        x = x @ self.w
        if self.use_bias: 
            x = x + self.b

        return x


class Flatten(Layer):
    def __init__(self):
        self.has_vars = False

    def __call__(self, x, training=False):

        return x.reshape((x.shape[0], -1))


class MaxPool2d(WeightedLayer):
    def __init__(self, dimensions):
        self.has_vars = False
        self.dimensions = dimensions

    def __call__(self, x, training=False):
        x = x if isinstance(x, (Tensor)) else Tensor(x)

        n, c, h, w = x.shape
        dh, dw = self.dimensions
        stride = dh
        h_out = h // dh         # TODO: modify for different stride?
        w_out = w // dw
        image = x.data

        assert dw == dh, 'not implemented: unequal maxpool strides'

        pooled = np.empty((n, c, w // stride, h // stride))

        ismax = np.zeros(x.shape)
        for nidx in range(n):
            for chan in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        patch = image[nidx, chan, stride * j:stride * j + stride, stride * i:stride * i + stride]
                        max = np.argmax(patch, axis=1)
                        pooled[nidx, chan, i, j] = patch[max[0], max[1]]
                        ismax[nidx, chan, stride * i + max[0], stride * j + max[1]] = 1.

        # convert `out` to a Tensor
        out = Tensor(pooled)

        def _backward():
            x.grad += np.ones_like(x.data) * ismax

        out._backward = _backward 
        return out


class BatchNorm(WeightedLayer):
    def __init__(self, input_dim):
        self.running_mean = 0
        self.running_var = 0
        self.n = 0
        self.has_vars = True
        self.w = ones((input_dim,))
        self.b = zeros((input_dim,)) 

    def __call__(self, x, training=False):
        self.n += 1
        if len(x.shape) == 4:

            mean = x.sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
            var = (x - mean)**2

            self.running_mean = (1 / self.n) * mean + ((self.n - 1) / self.n) * self.running_mean
            self.running_var = (1 / self.n) * var + (self.n - 1 / self.n) * self.running_var
            if training:

                x = (x - mean) / var
                x = self.w * x + self.b
            else:
                x = (x - self.running_mean) / self.running_var
        if len(x.shape) == 2:

            mean = x.sum(axis=(0, 1)) / (x.shape[0] * x.shape[1])
            var = (x - mean)**2

            self.running_mean = (1 / self.n) * mean + ((self.n - 1) / self.n) * self.running_mean
            self.running_var = (1 / self.n) * var + (self.n - 1 / self.n) * self.running_var

            if training:

                x = (x - mean) / var
                x = self.w * x + self.b
            else:
                x = (x - self.running_mean) / self.running_var

    def parameters(self):
        return [self.w, self.b]
