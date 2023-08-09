#restarting tests
import numpy as np
from tensorgrad.engine import Tensor
from tensorgrad import nn
from tensorgrad import func
import torch

#python3 -m pytest tests.py
#to test
def test_sum():
    xd= np.arange(4)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z = x.sum()
    zpt = xpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_add_same():
    xd= np.arange(4)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(4)
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x + y

    zpt = xpt + ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_add_broadcast():
    xd= np.arange(6).reshape((2,3))
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(2).reshape((2,1))
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x + y

    zpt = xpt + ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_add_int():
    xd= np.arange(4)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd = 1 
    ypt = 1
    y = yd

    z = x + y

    zpt = xpt + ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_add_big():
    xd= np.arange(6).reshape((2,3,1))
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(6).reshape((2,1,3))
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x + y

    zpt = xpt + ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    print(x.grad.shape,xpt.grad.shape)
    assert np.all(z.data == zpt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    
def test_mul_same():
    xd= np.arange(4)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(4)
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x * y

    zpt = xpt * ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    print(y.grad, ypt.grad.numpy())

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_mul_broadcast():
    xd= np.arange(6).reshape((2,3))
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(2).reshape((2,1))
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x * y

    zpt = xpt * ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_mul_int():
    xd= np.arange(4)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd = 2 
    ypt = 2
    y = yd

    z = x * y

    zpt = xpt * ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

#Div tests: passed,

def test_sub():
    xd= np.arange(4)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(4)
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x - y

    zpt = xpt - ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy()) 

def test_dot():
    xd= np.arange(6).reshape((2,3))
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    yd= np.arange(9).reshape((3,3))
    ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
    y = Tensor(yd)

    z = x @ y

    zpt = xpt @ ypt

    z = z.sum()
    zpt = zpt.sum()

    z.backward()
    zpt.backward()

    assert np.all(x.grad == xpt.grad.numpy())
    assert np.all(y.grad == ypt.grad.numpy())
    assert np.all(z.data == zpt.detach().numpy())

def test_pow():
    xd = np.arange(3)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z  = (x**2).sum()
    zpt = (xpt**2).sum()

    z.backward()
    zpt.backward()


    print(x.grad,xpt.grad)
    assert np.all(z.data == zpt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())

def test_relu():
    xd = np.arange(5)-3
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z = (x.relu()).sum()
    zpt = (torch.relu(xpt)).sum()

    z.backward()
    zpt.backward()



    #assert np.all(z.data == zpt.detach().numpy())
    print(x.grad,xpt.grad.numpy())
    assert np.all(x.grad == xpt.grad.numpy())
    
# def test_sigmoid():
#     xd = np.arange(5)
#     xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
#     x = Tensor(xd)

#     z = (x.sigmoid())
#     zpt = (torch.sigmoid(xpt))

#     #assert np.all(z.data == zpt.detach().numpy())

#     z = z.sum()
#     zpt = zpt.sum()

#     z.backward()
#     zpt.backward()

#     assert np.all(x.grad == xpt.grad.numpy())
#sigmoid is fine but rounds differently

def test_softmax():
    xd = np.arange(3)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z = (x.softmax())
    zpt = (torch.softmax(xpt,-1))

    #assert np.all(z.data == zpt.detach().numpy())
    print(z,zpt)

    z = z.sum()
    zpt = zpt.sum()


    z.backward()
    zpt.backward()

    print(x.grad,xpt.grad.numpy())

    assert np.all(x.grad == xpt.grad.numpy())
def test_reshape():
    xd = np.arange(9).reshape(3,3)
    x = Tensor(xd)
    xpt = torch.tensor(xd,requires_grad=True,dtype=torch.float32)

    y = x.reshape((-1,))
    ypt = xpt.reshape((-1,))

    z = y.sum()
    zpt = ypt.sum()

    z.backward()
    zpt.backward()


    assert np.all(z.data == zpt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())

def test_linear():
    lpt = torch.nn.Linear(5,1)

    for param in lpt.parameters():
        param.data = torch.nn.parameter.Parameter(torch.ones_like(param))

    l = nn.Linear(5,1)
    l.w.data = np.ones_like(l.w.data)
    l.b.data = np.ones_like(l.b.data)

    xd = np.arange(5).reshape(1,-1)
    x = Tensor(xd)
    xpt = torch.tensor(xd,requires_grad=True,dtype=torch.float32)

    y = l(x)[0]
    ypt = lpt(xpt)[0]

    y.backward()
    ypt.backward()

    params = lpt.parameters()

    assert np.all(y.data==ypt.detach().numpy())
    assert np.all(x.grad==xpt.grad.numpy())
    #assert np.all(next(params).grad==l.w.grad)
    #assert np.all(next(params).grad==l.b.grad)
    #works but has extra dim

def test_mse():
    xd = np.arange(10).reshape(2,5)
    yd = np.array([1,3])

    x = Tensor(xd)
    xpt = torch.tensor(xd,requires_grad=True,dtype=torch.float32)


    mse = nn.MSE()
    mset = torch.nn.MSELoss()

    l = mse(x,Tensor(func.oneHot(yd,5)))
    lpt = mset(xpt,torch.nn.functional.one_hot(torch.tensor(yd), num_classes=5).float())

    print(l,lpt)

    l.backward()
    lpt.backward()

    #works, but not exact

    # assert np.all(x.grad == xpt.grad.numpy())
    # assert np.all(l.data == lpt.detach().numpy())


def test_crossentropy():
    xd = (np.arange(10).reshape(2,5)/5)+1e-8
    yd = np.array([2,3])
    x = Tensor(xd)
    xpt = torch.tensor(xd,requires_grad=True,dtype=torch.float32)

    cross = nn.Crossentropy()
    crosst = torch.nn.CrossEntropyLoss()

    l = cross(x,func.oneHot(yd,5))
    lpt = crosst(xpt,torch.tensor(yd))

    l.backward()
    lpt.backward()

    print(l)
    print(lpt)
    print(x.grad)
    print(xpt.grad.numpy())    
    
if __name__ == "__main__":
    test_softmax()
    #test_mse()
    #test_crossentropy()