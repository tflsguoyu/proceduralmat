import torch as th
import numpy as np
from torchvision import *

def normalize_vgg19(input):
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.255]
    )
    return transform(input)

def inv_normalize_vgg19(input):
    transform = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    return transform(input)

def arr(*x):
	return th.tensor(x, dtype=th.float32)

def dstack(img, a):
	return th.stack([img * x for x in a], 2)

# def roll0(A, n):
# 	return th.cat((A[n:, :], A[:n, :]), 0)

# def roll1(A, n):
# 	return th.cat((A[:, n:], A[:, :n]), 1)

# def shift(A):
# 	m, n = A.shape
# 	assert m == n
# 	nh = n // 2
# 	tmp = roll0(A, nh)
# 	return roll1(tmp, nh)

def roll(dim, A, n):
    assert( A.dim() > dim );
    return th.cat((A.narrow(dim, n, A.size(dim) - n), A.narrow(dim, 0, n)), dim)

def roll0(A, n):
	return roll(0, A, n)

def roll1(A, n):
	return roll(1, A, n)

def shift(A):
    m = A.size(0)
    n = A.size(1)
    assert( m == n )
    nh = n // 2
    return roll1(roll0(A, nh), nh)

def normalize_all(x):
	s = th.sqrt((x ** 2).sum(2))
	return x / th.stack((s, s, s), 2)

def normals(hf, pix_size):
	c = 1 / pix_size
	dx = roll1(hf, 1) - hf
	dy = hf - roll0(hf, 1)
	N = th.stack((-c*dx, -c*dy, th.ones_like(dx)), 2)
	N = normalize_all(N)
	return N

def logit(u):
    return np.log(u/(1-u+1e-6)+1e-6)

def invLogit(v):
    return 1/(1+np.exp(-v))

def invLogitTensor(v):
    return 1/(1+(-v).exp())

def normTo01(x, a, b):
    return (x-a)/(b-a)
    
def invNormTo01(x, a, b):
    return x*(b-a)+a

##
def transVar(x, a, b):
	return logit(normTo01(x,a,b))

def invTransVar(y, a, b):
	return invNormTo01(invLogit(y),a,b)

def invTransVarTensor(y, a, b):
    return invNormTo01(invLogitTensor(y),a,b)

##
# def transVar(x, a, b):
#     return normTo01(x,a,b)

# def invTransVar(y, a, b):
#     return invNormTo01(y,a,b)

# def invTransVarTensor(y, a, b):
#     return invNormTo01(y,a,b)


##
# def invTransVarGrad(y, a, b):
# 	return (b-a) * invLogit(y) * (1- invLogit(y))

# def KL_div(a, b):
#     assert(np.shape(a) == np.shape(b))
#     h,w = np.shape(a)
#     kl_div = 0
#     epsilon = 0.000001
#     for i in range(h):
#         for j in range(w):
#             kl_div += a[i,j] * np.log(max(epsilon,a[i,j]) / max(epsilon,b[i,j]))
#     return kl_div

    