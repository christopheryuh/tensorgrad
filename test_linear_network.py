from tensorgrad import nn
from tensorgrad.models import Model
from tensorgrad import optimizers
from tensorgrad import func

import tensorflow as tf

import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape,y_train.shape)
print(x_train.shape,y_train.shape)

x_train = x_train/255.0


model = Model(
    [
        nn.Flatten(),
        nn.Linear(28*28,10),
        # nn.Relu(),
        # nn.Linear(14*14,7*7),
        # nn.Relu(),
        # nn.Linear(7*7,10),
        nn.Sigmoid()
    ]
)

optimizer = optimizers.SGD(model.parameters, lr=1e-4)

model.train(x_train,y_train,loss_fn=nn.MSE(),optimizer=optimizer,batch_size=100,label_depth=10,epochs=1)
