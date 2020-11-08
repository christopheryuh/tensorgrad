import numpy as np 

from tensorgrad.engine import *
from tensorgrad import utils
from tensorgrad.optimizers import *
from tensorgrad.func import *


epsilon = .00000000000001


class Activation():
    def __init__(self):
        self.has_vars = False


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


class Conv2d():
    def __init__(self, inputs, outputs, kernel_dim, use_bias=True, padding='valid'):

        assert padding in {'valid', 'same'}

        self.w = random(size=(outputs, inputs, kernel_dim, kernel_dim))
        self.outs = outputs
        self.use_bias = use_bias 
        self.padding_style = padding

        self.kh = kernel_dim
        self.kw = kernel_dim

        self.has_vars = True

        self.h2 = self.kh // 2
        self.w2 = self.kh // 2

        if self.use_bias:
            self.b = random(self.outs)

        def get_pixel_value(image, kernel, i, j):

            kh, kw = (kernel.shape[1], kernel.shape[2])

            h2 = kh // 2
            w2 = kw // 2

            patch = image[i - h2:i - h2 + kh, 
                          j - w2:j - w2 + kw]

            out = patch * kernel
            out = out.sum()

            return out

        self.get_pixel_value = get_pixel_value

        if padding == 'valid':

            def padding(image):
                return image

        if padding == 'same':

            def padding(image):
                out = zeros(image.shape[0] + self.h2, image.shape[1] + self.w2)
                out[h2:h2 + image.shape[0], w2:w2 + image.shape[1]] = image
                return out
        self.padding = padding

    def parameters(self):
        if self.use_bias:
            return [self.w, self.b]
        else:
            return [self.w, ]

    def __call__(self, x, training=False):

        x = x if isinstance(x, (Tensor)) else Tensor(x)

        image_shape = (x.shape[-2], x.shape[-1])

        if self.padding_style == 'valid':
            out = empty((self.outs, image_shape[0] - self.h2, image_shape[1] - self.w2))
        if self.padding_style == 'same':
            out = empty((self.outs, image_shape[0], image_shape[1]))

        for outidx in range(len(self.w.data)):
            for image in x:
                image = self.padding(image)

                for i in range(image.shape[0] - h2):
                    for j in range(image.shape[1] - h2):
                        out[outidx][i, j] = self.get_pixel_value(image, self.w[outidx])

        out = out.reshape((x.shape[0], self.outs, *image_shape))

        return out


class Linear():
    def __init__(self, n_in, n_out, use_bias=True):
        self.use_bias = use_bias
        self.has_vars = True
        self.w = random(size=(n_in, n_out))
        self.b = random(size=(1, n_out))

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


class Flatten():
    def __init__(self):
        self.has_vars = False

    def __call__(self, x, training=False):

        return x.reshape((x.shape[0], -1))


class MaxPool2d():
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
            x.grad += x.data * ismax

        out._backward = _backward 
        return out


class BatchNorm():
    def __init__(self):
        self.running_mean = 0
        self.running_var = 0
        self.n = 0
        self.has_vars = True
        self.w = random()
        self.b = random()

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
