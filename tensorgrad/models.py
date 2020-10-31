from tensorgrad.engine import *
from tensorgrad.func import *
from tensorgrad.utils import *
from tensorgrad.nn import *





class Model():
    def __init__(self,layers):
        self.layers = layers
        self._parameters = []
    def add(self,layer):
        self.layers = [*self.layers,layer]
    def __call__(self,x,training=False):
        for layer in self.layers:
            x = layer(x,training=True)
        return x
    def parameters(self):
        for layer in self.layers:
            if layer.has_vars == True:
                for param in layer.parameters():
                    self._parameters = [*self._parameters,param]
        return self._parameters
        
    def train(self,data,labels,optimizer=None,loss_fn=None,epochs=1,label_depth=None,update_after_epoch=True):
        assert loss_fn != None
        loss_fn = loss_fn()
        if optimizer == None:
            print("Warning: There is no optimizer set, default SGD is being used.")
            optimizer = SGD(self.parameters,lr=.001)
        if label_depth == None:
            print("Warning: Label depth set to 'None'. Please specify a label depth unless labels already one hot encoded.")
        for epoch in range(epochs):
            for x,y in zip(data,labels):
                x = x.reshape(-1,*x.shape)
                y = y.reshape(-1,1)
                if label_depth != None:
                    y = oneHot(y,depth=label_depth)
                y_hat = self(x)

                loss = loss_fn(y_hat,y)

                optimizer.zero_grad()

                optimizer.step()

            if update_after_epoch:
                print(f"loss:{loss}")
                
                
