from tensorgrad.engine import *
from tensorgrad.func import *
from tensorgrad.utils import *
from tensorgrad.nn import *
from matplotlib import pyplot as plt




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
        losslist = []
        
        print(labels.shape)

        if label_depth != None:
            labels = oneHot(labels.reshape(-1,1),depth=label_depth)

        for epoch in range(epochs):
            avg = []
            for x,y in zip(data,labels):
                x = x.reshape(-1,*x.shape)

                
                y_hat = self(x)

                loss = loss_fn(y_hat,y)

                optimizer.zero_grad()


                loss.backward()


                optimizer.step()

                avg.append(loss)

            if update_after_epoch:
                losslist.append(sum(avg[-y.shape[0]:])/y.shape[0])
                print('losslist',losslist)
                print(f"epoch:{epoch}/tloss:{sum(losslist[-y.shape[0]:])/y.shape[0]}")
        if update_after_epoch:
            plt.plot(range(len(losslist)),losslist)
            plt.show()

        data = np.array(data)
        
        pred = self(data[0])
        plt.imshow(data[0].reshape((28,28)))
        plt.show()
        print(np.argmax(pred.data))

        
                
                
