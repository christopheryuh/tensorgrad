from tensorgrad.engine import *
from tensorgrad.func import *
from tensorgrad.utils import *
from tensorgrad.nn import *
from matplotlib import pyplot as plt

import numpy as np




class Model():
    def __init__(self,layers):
        self.layers = layers
        self._parameters = []
    def add(self,layer):
        self.layers = [*self.layers,layer]  
    def __call__(self,x,training=False):
        x = x if isinstance(x, (Tensor)) else Tensor(x)
        #x = Tensor(np.random.uniform(0,1, size=x.shape))
        for num,layer in enumerate(self.layers):
            '''
            if len(np.array(x)[0,0].shape) < = 1:
                plt.imshow(np.array(x)[0,0].reshape(1,-1))
            else:
                plt.imshow(np.array(x)[0,0])
            plt.title('inputs' + layer.__class__.__name__)
            plt.show()
            '''
            #input(f"{layer.__class__.__name__}")
            x = layer(x,training=True)
            '''
            print(np.array(x)[0,0].shape)
            if len(np.array(x)[0,0].shape) <= 1:
                plt.imshow(np.array(x)[0,0].reshape(1,-1))
            else:
                plt.imshow(np.array(x)[0,0])    
            plt.title('outputs' + layer.__class__.__name__)
            plt.show()
            
            print(f"Got to layer number {num+1}")
            '''
            #if np.any(np.isnan(x.data)) or np.any(np.isinf(x.data)):
             #   print(num)      
             #   print('layer', layer.__class__.__name__)
                
        return x

    def parameters(self):
        for layer in self.layers:
            for param in layer.parameters():
                self._parameters = [*self._parameters,param]
        return self._parameters
        
    def train(self,data,labels,optimizer=None,loss_fn=None,epochs=1,batch_size=16,label_depth=None,update_after_epoch=True):
        assert loss_fn != None
        if optimizer == None:
            print("Warning: There is no optimizer set, default SGD is being used.")
            optimizer = SGD(self.parameters,lr=.001)
        if label_depth == None:
            print("Warning: Label depth set to 'None'. Please specify a label depth unless labels already one hot encoded.")
        losslist = []
        

        if label_depth != None:
            labels = oneHot(labels.reshape((-1,1)),depth=label_depth)

        

        for epoch in range(epochs):
            avg = []
            for i in range(0, data.shape[0], batch_size):

                x = data[i:i+batch_size]

                y = labels[i:i+batch_size]


                y_hat = self(x)


                loss = loss_fn(y_hat,y)


                optimizer.zero_grad()


                loss.backward()


                optimizer.step()

                avg.append(loss.data)

            if update_after_epoch:
                losslist.append(sum(losslist[-y.shape[0]:])/y.shape[0])
                print(f"epoch:{epoch}\t loss:{sum(losslist[-y.shape[0]:])/y.shape[0]}")
        if update_after_epoch:
            #print(losslist)
            plt.plot(range(len(losslist)),losslist)
            plt.show()

        data = np.array(data)
        
        pred = self(data[0])
        plt.imshow(data[0].reshape((28,28)))
        plt.show()
        print(np.argmax(pred.data))

        
                
                
