from tensorgrad import nn
from tensorgrad.models import Model
from tensorgrad import optimizers

import numpy as np

x = np.arange(50).reshape(-1,1)
y = (x*2)-5

model = Model(
    [
        nn.Linear(1,3),
        nn.Relu(),
        nn.Linear(3,1),
        
    ]
)

optimizer = optimizers.SGD(model.parameters, lr=.001)

model.train(x,y,loss_fn=nn.MSE(),optimizer=optimizer,epochs=100)

# for p in model.parameters():
#     print(p)