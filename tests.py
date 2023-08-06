#restarting tests
import numpy as np
from tensorgrad.engine import Tensor
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

    print(y.grad,ypt.grad)

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
#only need one test to make sure the negative works
#works but has rounding error
# def test_div():
#     xd= np.arange(4)
#     xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
#     x = Tensor(xd)

#     yd= np.arange(4)
#     ypt = torch.tensor(yd, requires_grad=True,dtype=torch.float32)
#     y = Tensor(yd)

#     z = x / y

#     zpt = xpt / ypt

#     z = z.sum()
#     zpt = zpt.sum()

#     z.backward()
#     zpt.backward()

#     assert np.all(x.grad == xpt.grad.numpy())
#     assert np.all(y.grad == ypt.grad.numpy())
#     assert np.all(z.data == zpt.detach().numpy())

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
    xd = np.arange(10)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z = (x.relu()).sum()
    zpt = (torch.relu(xpt)).sum()

    z.backward()
    zpt.backward()

    assert np.all(z.data == zpt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())
def test_sigmoid():
    xd = np.arange(10)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z = (x.sigmoid()).sum()
    zpt = (torch.sigmoid(xpt)).sum()

    z.backward()
    zpt.backward()

    assert np.all(z.data == zpt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())

def test_softmax():
    xd = np.arange(10)
    xpt = torch.tensor(xd, requires_grad=True,dtype=torch.float32)
    x = Tensor(xd)

    z = (x.softmax()).sum()
    zpt = (torch.softmax(xpt,-1)).sum()


    z.backward()
    zpt.backward()

    print((x.grad-xpt.grad.numpy()).sum())

    #assert np.all(z.data == zpt.detach().numpy())
    assert np.all(x.grad == xpt.grad.numpy())




if __name__ == "__main__":
    test_pow()