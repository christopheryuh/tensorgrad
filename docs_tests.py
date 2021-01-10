#---------Making the Dataset---------
from tensorgrad.engine import Tensor

x = Tensor([[1,0,0],[0,1,0],[1,1,1],[0,0,0],[1,1,0],[0,0,1]])

y = Tensor([1,0,1,0,1,0])

from tensorgrad import nn
from tensorgrad.models import Model


#---------Making The Model----------

model = Model([
    nn.Linear(3,9),
    nn.Relu(),
    nn.Linear(9,6),
    nn.Relu(),
    nn.Linear(6,1),
    nn.Sigmoid()

])

#---------Training The Model--------

from tensorgrad.optimizers import SGD

optimizer = SGD(model.parameters(),lr=0.1)


model.train(x, y, loss_fn=nn.Crossentropy(), label_depth=None,batch_size=1,optimizer=optimizer)
