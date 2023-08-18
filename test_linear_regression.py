from tensorgrad import nn
from tensorgrad.models import Model
from tensorgrad import optimizers

import numpy as np

x = np.arange(50).reshape(-1,1)+1
y = (x*2)-5
y = y/100

model = Model(
    [
        nn.Linear(1,3,use_bias=True),
        nn.Relu(),
        nn.Linear(3,1,use_bias=True),
        nn.Sigmoid(),
        
    ]
)

optimizer = optimizers.SGD(model.parameters, lr=1e-5)


model.train(x,y,loss_fn=nn.MSE(),optimizer=optimizer,epochs=10)