import torch as th
import numpy as np

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
    return np.log(u/(1-u))

def invLogit(v):
    return 1/(1+np.exp(-v))

def transVar(x, a, b):
	return logit((x-a)/(b-a))

def invTransVar(y, a, b):
	return a+(b-a)*invLogit(y)

def invTransVarGrad(y, a, b):
	return (b-a) * invLogit(y) * (1- invLogit(y))

