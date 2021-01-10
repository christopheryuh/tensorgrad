#---------Making the Dataset---------
from tensorgrad.engine import Tensor

x = Tensor([[.9,.1,.1],[.1,.9,.1],[.9,.9,.9],[.1,.1,.1],[.9,.9,0],[.1,.1,.9]])

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


model.train(x, y, loss_fn=nn.Crossentropy(), epochs=10,label_depth=None,batch_size=1,optimizer=optimizer)
