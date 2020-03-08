import torch as th
import numpy as np
from torchvision import *


def arr(*x):
	return th.tensor(x, dtype=th.float32)

def dstack(img, a):
	return th.stack([img * x for x in a], 2)



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
	s = th.sqrt((x ** 2).sum(2).clamp(min=1e-4))
	return x / th.stack((s, s, s), 2)

def normals(hf, pix_size):
	c = 1 / pix_size
	dx = roll1(hf, 1) - hf
	dy = hf - roll0(hf, 1)
	N = th.stack((-c*dx, -c*dy, th.ones_like(dx)), 2)
	N = normalize_all(N)
	return N


def from8bit(img):
	return th.tensor(np.asarray(img, dtype=np.float32) / 255)

def from16bit(img):
    return th.tensor(np.asarray(img, dtype=np.float32) / 65535)

def normal_lpdf(x, mu, sigma):
    tmp = (x - mu) / sigma
    return 0.5 * (mu**2).sum()

def sample_normal(mu, sigma, mn, mx, num=1):
    x = th.randn(num) * sigma + mu
    return x.clamp(mn, mx) # FIXME
