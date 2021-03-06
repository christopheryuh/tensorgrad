import torch
import numpy as np
from tensorgrad.engine import Tensor
from tensorgrad.func import *
from tensorgrad.nn import *
from tensorgrad.models import Model
from tensorgrad.optimizers import SGD


def helper(fn, fn_input):
    xpt = torch.Tensor(fn_input)
    xpt.requires_grad = True

    ypt = fn(xpt, pytorch=True)
    ypt.backward()

    x = Tensor(fn_input)

    y = fn(x)

    y.backward()

    assert np.all(y.data == ypt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())


def test_same_shape_add():
    fn_input = np.array([1, 2, 3])

    def fn(x, pytorch=False):
        if pytorch:      
            z = x + torch.tensor([4, 3, 2])
            return z.sum()
        else:
            z = x + np.array([4, 3, 2])
            return z.sum()
    helper(fn, fn_input)


def test_mul():
    fn_input = np.array([[1, 2], [1, 2]])

    def fn(x, pytorch=False):
        if pytorch:
            z = x * torch.tensor([[1, 2], [1, 2]])
            h = z * torch.tensor([1, 2])
            return h.sum()
        else:
            z = x * np.array([[1, 2], [1, 2]])
            h = z * np.array([1, 2])
            return h.sum()
    helper(fn, fn_input)


def test_broadcast_add():
    fn_input = np.array([1, 2, 3, 3])

    def fn(x, pytorch=False):
        if pytorch:
            z = fn_input + torch.tensor([[1], [2], [3]])
            return z.sum()
        else:
            z = fn_input + Tensor([[1], [2], [3]])
            return z.sum()
        helper(fn, fn_input)


def test_matmul():
    fn_input = np.array([[1., 2., 3., 3.], [1., 2., 3., 3.]])

    def fn(x, pytorch=False):
        if pytorch == True:
            z = x @ torch.tensor([[1], [2], [3], [3]], dtype=torch.float32)
            return z.sum()
        else:
            z = x @ np.array([[1], [2], [3], [3]])
            return z.sum()
    helper(fn, fn_input)


'''
test_same_shape_add()
test_mul()
test_broadcast_add()
test_matmul()
'''


def make_a_linear_layer():
    x = np.array([[1], [1], [1]]).astype(np.float32)
    layer = Linear(3, 5)
    y = layer(x)
    print('output', y)
    z = y.sum()
    z.backward()
    print('with grad', layer.w, layer.b)
    layer.zero_grad()
    print('zero grad', layer.w, layer.b)


def test_setitem():
    print('Testing setitem')
    x = Tensor([1, 2, 3, 4])
    z = 1
    x[0] = z

    w = x.sum()
    w.backward()

    xpt = torch.tensor([1., 2., 3., 4.])
    xpt.requires_grad = True

    zpt = 1.

    xpt[0] = zpt

    wpt = x.sum()
    wpt.backward()

    print(xpt.grad)

    assert np.all(w.data == wpt.data)
    assert np.all(x.grad == xpt.grad)


# make_a_linear_layer()
# test_setitem()


def make_a_conv2d_layer():
    x = np.ones((1, 1, 5, 5))
    layer = Conv2d(1, 3, 3, use_bias=False, padding='same')

    out = layer(x)

    print(out)
    print(out.shape)

    print('going backward')
    z = out.sum()
    z.backward()
    print(z)
    print(layer.w.grad)


# make_a_conv2d_layer()
print('--maxpool--')


def make_max_pooling():
    maxp = MaxPool2d((2, 2))
    x = Tensor(np.array([[[1, 2, 3, 4],
                          [1, 2, 3, 4],
                          [1, 2, 3, 4],
                          [1, 2, 3, 4]]]).reshape(1, 1, 4, 4))

    y = maxp(x)

    z = y.sum()
    print(y, x) 
    z.backward()
    print(x.grad)


#make_max_pooling()
#make_a_conv2d_layer()

print("-------Testing Conv-Net----------")


def make_a_convnet():

    from tensorflow.keras.datasets import mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(-1, 1, 28, 28)/255.0

    print('got data')
    print(train_labels.shape)

    convnet = Model([])
    convnet.add(Conv2d(1, 8, 3, padding='same'))
    convnet.add(Relu())
    convnet.add(MaxPool2d((2, 2)))
    convnet.add(Conv2d(8, 8, 3, padding='same'))
    convnet.add(Relu())
    convnet.add(MaxPool2d((2, 2)))
    convnet.add(Conv2d(8, 8, 3, padding='same'))
    convnet.add(Flatten())
    convnet.add(Linear(8 * 7 * 7, 10))
    convnet.add(Softmax())

    print(len(convnet.parameters()))

    print(train_images.shape)
    print(train_labels.shape)

    optimizer = SGD(convnet.parameters(), lr=0.1)

    convnet.train(
        train_images, train_labels,
        loss_fn=Crossentropy(), label_depth=10, batch_size=100,
        optimizer=optimizer,
        epochs=5)



#make_a_convnet()

print("---- testing linear training --- ")

def test_linear_training():
    tglin = Linear(3,1,use_bias=False)

    ptlin = torch.nn.Linear(3,1,bias=False)

    ptlin.weight = torch.nn.Parameter(torch.Tensor(np.array(tglin.w)))

    print(tglin.w)
    print(ptlin.weight)
    
    xg = np.array([[1,1,1]])

    yg = np.array([1])

    pto = torch.optim.SGD([ptlin.weight], .01, .9)
    tgo = SGD([tglin.w],lr=.01)

    xtg = Tensor(xg)

    xpt = torch.Tensor(xg)

    y_hattg = tglin(xtg)

    lossfntg = Crossentropy()

    losstg = lossfntg(y_hattg, Tensor(yg))
    
    print(xpt, ptlin.weight)
    print(xpt.shape, ptlin.weight.shape)

    y_hatpt = ptlin(xpt)

    lossfnpt = torch.nn.CrossEntropyLoss()

    losspt = lossfnpt(y_hatpt, torch.Tensor(yg))

    assert np.isclose(losstg.numpy, losspt.numpy)





test_linear_training()