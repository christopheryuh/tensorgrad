# Tensorgrad

## Basic math with autograd


``` { .py }

from tensorgrad.engine import Tensor

a = Tensor([1,2,3])

b = Tensor([1,2,3])

c = Tensor([1,2,3])

x = a + b

y = x + c

z = y.sum()

print("Before Gradient Calculation")
print(a,b,c,x,y)


z.backward()

print("After gradient Calculation")
print(a,b,c,x,y)


```


## Linear Regression with the Model class 

``` { .py }
from tensorgrad.models import Model
from tensorgrad.optimizers import SGD
from tensorgrad.nn import Linear
from tensorgrad.nn import MSE

data = Tensor([[1],[2],[3],[4])
labels = Tensor([2,4,6,8])

model = Model()

model.add(Linear(1,1))

optim = SGD(model.parameters(), lr=.01)

model.train(data, labels, optimizer=optim, loss_fn=MSE, epochs = 100, batch_size=4)

```

First we implement all that is needed (1-4), then we create our training data, and the corresponding labels (5-6).
After that we initialize our model and add a Linear layer with 1 input and 1 output(this specific layer would be the same as "y=mx+b") (8-10).
Once we have that, we can construct our optimizer with a learning rate of .01. 
With everything created, we can finally train our model. Here we give "model.train" training data and labels, as well as an optimizer and a loss function (to determine the error of the model), we also give 100 epochs, or rounds of training. We also specify a batch size of 4 to save time.	 


## Basic Neural Network


``` { .py }
from tensorgrad.models import Model
from tensorgrad.optimizers import SGD
from tensorgrad.nn import Linear, Relu
from tensorgrad.nn import MSE

data = Tensor([[1],[2],[3],[4])
labels = Tensor([2,4,6,8])

model = Model()

model.add(Linear(1,5))
model.add(Relu())
model.add(Linear(5,5))
model.add(Relu())
model.add(Linear(5,1))
model.add(Relu())

optim = SGD(model.parameters(), lr=.01)

model.train(data, labels, optimizer=optim, loss_fn=MSE, epochs = 100, batch_size=4)

```
Here we do almost the exact same thing as previously, with the addition of two extra Linear layers and activation functions. Adding these layers actually makes this model much more complex. This model suddenly goes from 1 parameter to 35. We also add activation functions, or nonlinearities. In this model, these functions serve mainly to divide the linear layers, stopping them from merging to become 1 linear layer (essentially stoping the communitive property). 

## Convolutions

Currently working out errors with Convolution Neural Networks. Check back soon!