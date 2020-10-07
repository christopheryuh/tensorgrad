import numpy as np 

from tensorgrad.engine import random,empty,zeros,Tensor


def oneHot(x,depth=None):
    assert (depth != None)
    zeros = np.zeros(x.shape[0],depth)
    for idx,i in enumerate(x):
        zeros[idx][i] = 1
    return zeros


        


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
    def __init__(self,inputs,outputs,kernel_dim,use_bias=True,padding='valid',nonlin='relu'):

        assert padding in {'valid', 'same'}

        self.w = random(size=(outputs,inputs,kernel_dim,kernel_dim))
        self.outs = outputs
        self.use_bias = use_bias 
        self.nonlin = nonlin
        self.padding_style = padding

        self.kh = kernel_dim
        self.kw = kernel_dim

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
            return [sefl.w,]



    def __call__(self,x):

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




        if self.use_bias:
            out = out + self.b
        
        if self.nonlin == None:
            return x
        if self.nonlin == 'relu':
            x = x.relu()
        elif self.nonlin == 'sigmoid':
            x = x.sigmoid()


        return x

        return out



class Linear(Module):
    def __init__(self,n_in,n_out,use_bias = True,nonlin='relu'):
        self.use_bias = use_bias
        self.nonlin = 'relu'
        self.w = random(size=(n_out,n_in,))
        self.b = random(size=(n_out,1))
    def parameters(self):
        if self.use_bias:
            return [self.w,self.b]
        else:
            return [self.w,]
    def __call__(self,x):
        x = x if isinstance(x,(Tensor)) else Tensor(x)
        x = self.w @ x
        if self.use_bias:
            x = x + self.b
        if self.nonlin == None:
            return x
        if self.nonlin == 'relu':
            x = x.relu()
        elif self.nonlin == 'sigmoid':
            x = x.sigmoid()
        elif self.nonlin == 'softmax':
            x = x.softmax()


        return x


class Flatten():
    def __call__(self,x):

        return x.reshape((x.shape[0],-1))

class MaxPool2d():
    def __init__(self,dimesions):
        self.dimensions = dimesions
        
    def __call__(self,x):
        x = x if isinstance(x,(Tensor)) else Tensor(x)
        image = x.data
        out = np.empty((x.shape[0],x.shape[1]//self.dimensions[0],x.shape[2]//self.dimensions[1]))
        ismax = zeros(x.shape)
        for row in range(image.shape[2]//self.dimensions[0]):
            for col in range(image.shape[3]//self.dimensions[1]):
                ismax[:,:,row:row+self.dimensions[0],col:col+self.dimensions[1]] = (

                    image[:,:,row:row+self.dimensions[0],col:col+self.dimensions[1]] == 
                    image[:,:,row:row+self.dimensions[0],col:col+self.dimensions[1]].max(axis=(2,3),keepdims=True)

                )

                out[:,:,row,col] = image[:,:,row:row+self.dimensions[0],col:col+self.dimensions[1]].max(axis=(2,3),keepdims=True)


        def _backward():
            x.grad += (ismax  * x.data) * out.grad


        out._backward = _backward 
        return out





class Model(Module):
    def __init__(self,layers):
        self.layers = layers
        self.parameters = self.layers
    def add(self,layer):
        self.layers = [*self.layers,layers]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        for layer in self.layers:
            self.parameters = [*self.parameters,layer]
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
                
                y_hat = self(data)

                loss = lossFn(y_hat,y)
                
                loss.backward()

                optimizer.step()
            print(f"epoch:{epoch + 1}\tloss:{loss}")