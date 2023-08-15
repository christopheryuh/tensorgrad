from tensorgrad import nn
from tensorgrad.models import Model
from tensorgrad import optimizers

import numpy as np

x = np.arange(1).reshape(-1,1)+1
y = (x*2)-5
y = y/100

model = Model(
    [
        nn.Linear(1,1,use_bias=True),
        #nn.Relu(),
        
    ]
)

optimizer = optimizers.SGD(model.parameters, lr=1e-4)

b = []


model.train(x,y,loss_fn=nn.MSE(),optimizer=optimizer,epochs=10)

for p in model.parameters():
    print(p)

# m2 = Model([nn.Sigmoid()])
# x = np.arange(20)-10.
# y = 1 / (1 + np.exp(-x))
# z = y*(1-y)

# import matplotlib.pyplot as plt

# plt.plot(x,y)
# plt.plot(x,z)
# plt.show()