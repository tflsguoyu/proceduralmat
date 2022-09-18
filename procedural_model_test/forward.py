import torch as th
import numpy as np
from PIL import Image
from lookup import *

eps = 1e-4

def from8bit(img):
    return th.tensor(np.asarray(img, dtype=np.float32) / 255)

def from16bit(img):
    return th.tensor(np.asarray(img, dtype=np.float32) / 65535)

def p2x(p, n):
    return (p*2 + 1) / n - 1
def x2p(x, n):
    p = ((1+x) * n - 1) / 2
    return p%n

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

class Material:
    def __init__(self, imres, device):
        self.device = device
        self.n = imres
        self.size = 25
        self.camera = 25
        self.initGeometry()

    def loadPara(self, para):
        self.para = para

    def sample_normal(self, mu, sigma, mn, mx, num=1):
        x = th.randn(num) * sigma + mu
        return x.clamp(mn, mx)
        
    def sample_prior(self):
        xs = []
        for p in self.paraPr:
            x = self.sample_normal(p[0].item(), p[1].item(), p[2].item(), p[3].item())
            xs.append(x)
        return np.hstack(xs)

    def initGeometry(self):
        # surface positions
        v = th.arange(self.n, dtype=th.float32, device=self.device)
        v = ((v + 0.5) / self.n - 0.5) * self.size
        y, x = th.meshgrid((v, v))
        pos = th.stack((x, -y, th.zeros_like(x)), 2)
        self.pos_norm = pos.norm(2.0, 2)

        # directions (omega_in = omega_out = half)
        self.omega = th.tensor([0,0,self.camera], dtype=th.float32, device=self.device) - pos
        self.dist_sq = self.omega.pow(2.0).sum(2).clamp(min=eps)
        d = self.dist_sq.sqrt()
        self.omega /= th.stack((d, d, d), 2)

        normal_planar = th.zeros_like(self.omega)
        normal_planar[:,:,2] = 1

        self.geom_planar, self.n_dot_h_planar = self.computeGeomTerm(normal_planar)

    def computeGeomTerm(self, normal):
        n_dot_h = (self.omega * normal).sum(2).clamp(eps,1)
        geom = n_dot_h / self.dist_sq

        return geom, n_dot_h

    def ggx_ndf(self, cos_h, alpha):
        denom = np.pi * alpha**2 * ((1 - cos_h**2) / ((alpha)**2).clamp(eps,1) + cos_h**2)**2
        return 1.0 / denom.clamp(min=eps)

    def brdf(self, n_dot_h, alpha, f0):
        D = self.ggx_ndf(n_dot_h, alpha)
        return f0 * D / (4 * (n_dot_h**2).clamp(eps,1))

    def logit(self, u):
        u = u.clamp(eps, 1-eps)
        return (u/(1-u)).log()

    def sigmoid(self, v):
        return 1/(1+(-v).exp())

    def ab_to_one(self, u, a, b):
        return (u-a)/(b-a)

    def one_to_ab(self, v, a, b):
        return a+(b-a)*v

    def orig_to_norm(self, para):

        y = self.logit(self.ab_to_one(para[self.paraId], 
                                      self.paraPr[self.paraId, 2], 
                                      self.paraPr[self.paraId, 3]))
        return y

    def norm_to_orig(self, y):

        x = self.one_to_ab(self.sigmoid(y), 
                           self.paraPr[self.paraId, 2], 
                           self.paraPr[self.paraId, 3])
        return x

    def unpack(self, y):
        para = th.tensor(self.para, dtype=th.float32, device=self.device)

        if y is not None:
            x = self.norm_to_orig(y)
            para[self.paraId] = x

        i = 0
        paraList = []
        for d in self.paraCh:
            paraList.append(para[i:i+d])
            i += d
        
        return paraList

    def eval_prior_lpdf(self, y):
        sigmoid_y = self.sigmoid(y)
        x = self.one_to_ab(sigmoid_y, 
                           self.paraPr[self.paraId, 2], 
                           self.paraPr[self.paraId, 3])
        x = (x - self.paraPr[self.paraId, 0]) / self.paraPr[self.paraId, 1]
        lpdf = (0.5*x**2).sum()

        sigmoid_y = sigmoid_y.clamp(1e-4, 1-1e-4)
        lpdf -= (sigmoid_y * (1-sigmoid_y)).log().sum()

        return lpdf