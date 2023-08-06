from tensorgrad.engine import Tensor
from tensorgrad.func import *
from tensorgrad.utils import *
from tensorgrad.nn import *
from matplotlib import pyplot as plt

import numpy as np


class Model():
    def __init__(self, layers):
        self.layers = layers

    def add(self, layer):
        self.layers = [*self.layers, layer]  

    def __call__(self, x, training=False):
        x = x if isinstance(x, (Tensor)) else Tensor(x)
        for num, layer in enumerate(self.layers):
            x = layer(x, training=True)

        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            for param in layer.parameters():
                params.append(param)
        return params

    def train(self, data, labels, optimizer=None, loss_fn=None, epochs=1, batch_size=16,
              label_depth=None, update_after_epoch=True):
        assert loss_fn is not None

        if optimizer is None:
            print("Warning: There is no optimizer set, default SGD is being used.")
            optimizer = SGD(self.parameters, lr=.1)
        if label_depth is None:
            print("Warning: Label depth set to 'None'. Please specify a label depth unless labels already one hot encoded.")

        losslist = []

        if label_depth is not None:
            labels = oneHot(labels.reshape((-1, 1)), depth=label_depth)
        loss = Tensor(0)
        for epoch in range(epochs):
            avg = []
            for t, i in enumerate(range(0, data.shape[0], batch_size)):
                if t % 10 == 0:
                    print(f'iter {i} / {data.shape[0]}')
                    print(loss.data)
                x = data[i:i + batch_size]

                y = labels[i:i + batch_size]

                y_hat = self(x)

                loss = loss_fn(y_hat, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                losslist.append(loss.data)

                avg.append(loss.data)

            if update_after_epoch:
                losslist.append(sum(avg[-y.shape[0]:]) / y.shape[0])
                # print('losslist', losslist)
                print(f"epoch:{epoch}/tloss:{sum(losslist[-y.shape[0]:])/y.shape[0]}")

        data = np.array(avg)
        plt.plot(range(len(avg)), avg)
        plt.show()