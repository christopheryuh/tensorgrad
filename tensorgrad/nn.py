import numpy as np 

from tensorgrad.engine import random,empty,zeros,Tensor


def oneHot(x,depth=None):
    assert (depth != None)
    zeros = np.zeros(x.shape[0],depth)
    for idx,i in enumerate(x):
        zeros[idx][i] = 1
    return zeros




class Relu():
    def __call__(self,x):
        return x.relu()

class Sigmoid():
    def __call__(self,x):
        return x.sigmoid()

class Softmax():
    def __call__(self,x):
        return x.softmax()
        
    


        


class Crossentropy():
    def __call__(self,y_hat,y):
        labels = oneHot(y,depth=y_hat.shape[-1])
        
        out = Tensor(-1*y_hat*np.log(labels))
        out = out.sum()

        def _backward():
            y_hat.grad += -1*labels**-1

        out._backward = _backward
        
        return out
        


class Optimizer():
    def __init__(self,parameters,lr=.001):
        self.lr = lr
        self.parameters = parameters


class SGD(Optimizer):
    def step(self):
        for param in self.parameters:
            param.data = param.data - (param.grad * self.lr)
            





class Module():
    def zero_grad(self):
        for p in self.parameters():
            p.grad[:] = 0.



class Conv2d(Module):
    def __init__(self,inputs,outputs,kernel_dim,use_bias=True,padding='valid'):

        assert padding in {'valid', 'same'}

        self.w = random(size=(outputs,inputs,kernel_dim,kernel_dim))
        self.outs = outputs
        self.use_bias = use_bias 
        self.padding_style = padding

        self.kh = kernel_dim
        self.kw = kernel_dim


        self.has_vars = True

        self.h2 = self.kh//2
        self.w2 = self.kh//2

    
        if self.use_bias:
            self.b = random(self.outs)
        


        def get_pixel_value(image,kernel,i,j):

            kh,kw = (kernel.shape[1],kernel.shape[2])

            h2 = kh // 2
            w2 = kw // 2

            patch = image[i - h2:i - h2 + kh, 
                       j - w2:j - w2 + kw]
            
            out = patch*kernel
            out = out.sum()

            return out

        self.get_pixel_value = get_pixel_value

        if padding == 'valid':


            def padding(image):
                return image


        if padding == 'same':

            def padding(image):
                out = zeros(image.shape[0]+self.h2,image.shape[1]+self.w2)
                out[h2:h2+image.shape[0],w2:w2+image.shape[1]] = image
                return out
        self.padding = padding

    def parameters(self):
        if self.use_bias:
            return [self.w,self.b]
        else:
            return [self.w,]



    def __call__(self,x,training=False):


        assert len(x.shape) == 4
        print(x.shape)

        x = x if isinstance(x,(Tensor)) else Tensor(x)
        
        image_shape = (x.shape[-2],x.shape[-1])


        if self.padding_style == 'valid':
            out = empty((self.outs,image_shape[0]-self.h2,image_shape[1]-self.w2))
        if self.padding_style == 'same':
            out = empty((self.outs,image_shape[0],image_shape[1]))
        
        for outidx in range(len(self.w.data)):
            for image in x:
                image = self.padding(image)
                
                for i in range(image.shape[0]-h2):
                    for j in range(image.shape[1]-h2):
                        out[outidx][i,j] = self.get_pixel_value(image,self.w[outidx])




        return x

        return out




class Linear(Module):
    def __init__(self,n_in,n_out,use_bias = True):
        self.use_bias = use_bias
        self.has_vars = True
        self.w = random(size=(n_out,n_in,))
        self.b = random(size=(n_out,1))
    def parameters(self):
        if self.use_bias:
            return [self.w,self.b]
        else:
            return [self.w,]
    def __call__(self,x,training=False):
        x = x if isinstance(x,(Tensor)) else Tensor(x)
        x = self.w @ x
        if self.use_bias:
            x = x + self.b


        return x


class Flatten():
    def __call__(self,x):

        return x.reshape((x.shape[0],-1))

class MaxPool2d():
    def __init__(self,dimensions):
        self.dimensions = dimensions
        
    def __call__(self,x):
        x = x if isinstance(x,(Tensor)) else Tensor(x)

        image = x.data
        out = np.empty((x.shape[0], x.shape[1], x.shape[2] // self.dimensions[0], x.shape[3] // self.dimensions[1]))
        ismax = np.zeros_like(image, dtype=np.bool)

        for i, oi in zip(range(0, image.shape[2], self.dimensions[0]),
                         range(0, out.shape[2], 1)):
            for j, oj in zip(range(0, image.shape[3], self.dimensions[1]),
                             range(0, out.shape[3], 1)):
                print(i, j, oi, oj)

                # get the maximum in the window: shape = (N, C, 1, 1)
                flat_window = image[:, :, i:i + self.dimensions[0], j:j + self.dimensions[1]].reshape(image.shape[0], image.shape[1], -1)  # N, C, dim[0] * dim[1]
                index_array = np.argmax(flat_window, axis=-1)
                indices = np.unravel_index(index_array, self.dimensions)
                indices = np.array(indices).reshape(-1,)
                print(flat_window)
                print(indices)

                # set the booleans for the window in ismax: shape = (N, C, dim[0], dim[1])
                ismax[:,:,indices] = True
                # ismax[:, :, i:i + self.dimensions[0], j:j + self.dimensions[0]] = (
                #     image[:, :, i:i + self.dimensions[0], j:j + self.dimensions[1]] == maxval)

                # set the single pixel in the output: shape = (N, C, 1, 1)
                print('window with indexing',flat_window[:][:][indices[0]][indices[1]])
                out[:, :, oi, oj] = flat_window[0][indices[0]][indices[1]]


        # convert `out` to a Tensor
        out = Tensor(out)

        def _backward():
            # add the gradient from the output of the chosen pixel
            print(x.grad.shape, ismax.shape, out.grad.shape)
            print(x.grad[ismax])
            print(ismax)
            x.grad[ismax] += out.grad

        out._backward = _backward 
        return out


class Model(Module):
    def __init__(self,layers):
        self.layers = layers
        self.parameters = self.layers
    def add(self,layer):
        self.layers = [*self.layers,layers]
    def __call__(self,x,training=False):
        for layer in self.layers:
            x = layer(x,training=True)
        return x
    def parameters(self):
        for layer in self.layers:
            if layer.has_vars == True:
                self.parameters = [*self.parameters,layer.parameters()]
        return self.parameters
    def train(self,x,y,optimizer=None,lossFn=None,epochs=1):
        if optimizer == None:
            opt = SGD(self.parameters)
        else:
            opt = optimizer
        if lossFn == None:
            lossFn = Crossentropy()

        for epoch in range(epoch):
            for batch in zip(x,y):
                data, labels = batch
                
                y_hat = self(data,training=True)

                loss = lossFn(y_hat,y)
                
                loss.backward()

                optimizer.step()
            print(f"epoch:{epoch + 1}\tloss:{loss}")


class BatchNorm(Module):
    def __init__(self):
        self.running_mean = 0
        self.running_var = 0
        self.n = 0
        self.has_vars = True
        self.w = random()
        self.b = random()
    def __call__(self,x,training=False):
        self.n += 1
        if len(x.shape) == 4:

            mean = x.sum(axis=(0,2,3))/(x.shape[0]*x.shape[2]*x.shape[3])
            var = (x - mean)**2

            self.running_mean = (1/self.n)*mean + ((self.n-1)/self.n)*self.running_mean
            self.running_var = (1/self.n) * var + (self.n-1/self.n)*self.running_var
            if training:

                x = (x - mean)/var
                x = self.w*x + self.b
            else:
                x = (x - self.running_mean)/self.running_var
        if len(x.shape) == 2:

            mean = x.sum(axis=(0,1))/(x.shape[0]*x.shape[1])
            var = (x - mean)**2

            self.running_mean = (1/self.n)*mean + ((self.n-1)/self.n)*self.running_mean
            self.running_var = (1/self.n) * var + (self.n-1/self.n)*self.running_var

            if training:

                x = (x-mean)/var
                x = self.w*x + self.b
            else:
                x = (x - self.running_mean)/self.running_var
    def parameters(self):
        return [w,b]

                
