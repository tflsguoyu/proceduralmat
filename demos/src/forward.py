import torch as th
import numpy as np
import matplotlib.pyplot as plt
from gytools import *
from lookup import *
import exr

eps = 1e-4

def p2x(p, n):
    return (p*2 + 1) / n - 1
def x2p(x, n):
    p = ((1+x) * n - 1) / 2
    return p%n

class Material:
    def __init__(self, args, isHDR, device):
        self.device = device
        self.dir = args.out_dir
        self.n = args.imres
        self.isHDR = isHDR
        if isHDR:
            print('Target is a HDR image!!')
        self.size = args.size
        self.camera = args.camera
        self.initGeometry()

    def loadPara(self, para, paraId):
        self.para = para
        self.paraId = paraId
        self.plotPrior()
        # exit()

    def sample_prior(self):
        xs = []
        for p in self.paraPr:
            x = sample_normal(p[0].item(), p[1].item(), p[2].item(), p[3].item())
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

        # non-correct prior
        # lpdf = (0.5*y**2).sum()

        return lpdf

    def plotPrior(self):
        def gaussian(x, mu, sigma):
            return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x-mu)/sigma, 2.)/2)

        N = len(self.paraCh)
        rows = 1
        cols = N

        if N > 6:
            rows = int((N-0.1)//6) + 1
            cols = int(np.ceil(N/rows))
        fig = plt.figure(figsize=(2.56*cols,2.56*rows))
        for i in range(N):
            r = np.sum(self.paraCh[:i+1])-1
            p = self.paraPr[r,:].cpu().numpy()
            plt.subplot(rows,cols,i+1)
            x = np.linspace(p[2], p[3], 1000)
            y = gaussian(x, p[0], p[1])
            plt.plot(x, y)
            # plt.plot([self.para[r], self.para[r]], [0, max(y)*1.1])
            # plt.title(self.paraNm[i] + ' (init: %.3f)' % self.para[r])
            plt.title(self.paraNm[i])
        for a in fig.axes:
            a.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(self.dir + 'prior.png')
        plt.close()




# def lookupXd(A, X):   
#     # A.size = [m,m,...,m,c]; A.ndim = d+1
#     # X.size = [d,n,n,...,n]; X.ndim = d+1
#     # B.size = [n,n,...,n,c]; B.ndim = d+1
#     assert(A.dim() > 1 and X.dim() > 1)

#     n = A.size()[0]
#     c = A.size()[-1]
#     d = A.dim() - 1
#     n0 = X.size()[-1]

#     assert(X.size()[0] == d)

#     # print('This is a %dD data with size of %d^%d, and %d channel(s) for each data point' % (d, n, d, c))
#     # print('will lookup a subset with size of %d^%d' % (n0, d))

#     P = x2p(X, n)
#     P_int = P.floor() 
#     W = P - P_int
#     P_int = P_int.type(th.int64)
    
#     Idx = th.zeros_like(P_int)
#     B = 0

#     for i in range(2**d):
#         w = 1
#         for j in range(d):
#             if (i & (1<<j)) > 0:
#                 Idx[j] = P_int[j]+1
#                 w *= W[j]
#             else:
#                 Idx[j] = P_int[j]
#                 w *= (1- W[j])

#         if d == 1:
#             Bi = A[Idx[0]]
#         elif d == 2:
#             Bi = A[Idx[0], Idx[1]]
#         elif d == 3:
#             Bi = A[Idx[0], Idx[1], Idx[2]]

#         B += Bi * w.unsqueeze(-1).expand_as(Bi)

#     return B

# class ResizedCenterCrop(th.autograd.Function):

#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, I0, s, n):
#         s = s.detach()
#         # print('s:', s)
#         ctx.save_for_backward(s)
#         ctx.I0 = I0
#         ctx.ndim = ctx.I0.dim()
#         ctx.n0 = I0.size()[0]
        
#         v = th.arange(n, dtype=th.float32, device=I0.device)
#         i, j = th.meshgrid((v, v))
#         P = th.stack([i,j], 2)
#         ctx.x = j2x(j,n)
#         ctx.y = i2y(i,n)
#         ctx.i0 = y2i(ctx.y*s, ctx.n0)
#         ctx.j0 = x2j(ctx.x*s, ctx.n0)
#         ctx.i0_int = ctx.i0.floor()
#         ctx.j0_int = ctx.j0.floor()

#         I1 = ctx.I0[ctx.i0_int.type(th.int64), ctx.j0_int.type(th.int64)]
#         w1 = (1-(ctx.i0-ctx.i0_int)) * (1-(ctx.j0-ctx.j0_int))
#         I2 = ctx.I0[ctx.i0_int.type(th.int64), ctx.j0_int.type(th.int64)+1]
#         w2 = (1-(ctx.i0-ctx.i0_int)) * (ctx.j0-ctx.j0_int)
#         I3 = ctx.I0[ctx.i0_int.type(th.int64)+1, ctx.j0_int.type(th.int64)]
#         w3 = (ctx.i0-ctx.i0_int) * (1-(ctx.j0-ctx.j0_int))
#         I4 = ctx.I0[ctx.i0_int.type(th.int64)+1, ctx.j0_int.type(th.int64)+1]
#         w4 = (ctx.i0-ctx.i0_int) * (ctx.j0-ctx.j0_int)

#         if ctx.ndim == 3:
#             w1 = w1.unsqueeze(2).expand_as(I1)
#             w2 = w2.unsqueeze(2).expand_as(I2)
#             w3 = w3.unsqueeze(2).expand_as(I3)
#             w4 = w4.unsqueeze(2).expand_as(I4)

#         I = I1 * w1 + I2 * w2 + I3 * w3 + I4 * w4
#         return I

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         s, = ctx.saved_tensors

#         Dx1 = ctx.I0[ctx.i0_int.type(th.int64), ctx.j0_int.type(th.int64)]
#         wx1 = (1-(ctx.i0-ctx.i0_int)) * (-0.5*ctx.n0)
#         Dx2 = ctx.I0[ctx.i0_int.type(th.int64), ctx.j0_int.type(th.int64)+1]
#         wx2 = (1-(ctx.i0-ctx.i0_int)) * ( 0.5*ctx.n0)
#         Dx3 = ctx.I0[ctx.i0_int.type(th.int64)+1, ctx.j0_int.type(th.int64)]
#         wx3 = (ctx.i0-ctx.i0_int) * (-0.5*ctx.n0)
#         Dx4 = ctx.I0[ctx.i0_int.type(th.int64)+1, ctx.j0_int.type(th.int64)+1]
#         wx4 = (ctx.i0-ctx.i0_int) * ( 0.5*ctx.n0)

#         if ctx.ndim == 3:
#             wx1 = wx1.unsqueeze(2).expand_as(Dx1)
#             wx2 = wx2.unsqueeze(2).expand_as(Dx2)
#             wx3 = wx3.unsqueeze(2).expand_as(Dx3)
#             wx4 = wx4.unsqueeze(2).expand_as(Dx4)

#         df0_dx0 = Dx1 * wx1 + Dx2 * wx2 + Dx3 * wx3 + Dx4 * wx4  

#         Dy1 = ctx.I0[ctx.i0_int.type(th.int64), ctx.j0_int.type(th.int64)]
#         wy1 = ( 0.5*ctx.n0) * (1-(ctx.j0-ctx.j0_int))
#         Dy2 = ctx.I0[ctx.i0_int.type(th.int64), ctx.j0_int.type(th.int64)+1]
#         wy2 = ( 0.5*ctx.n0) * (ctx.j0-ctx.j0_int)
#         Dy3 = ctx.I0[ctx.i0_int.type(th.int64)+1, ctx.j0_int.type(th.int64)]
#         wy3 = (-0.5*ctx.n0) * (1-(ctx.j0-ctx.j0_int))
#         Dy4 = ctx.I0[ctx.i0_int.type(th.int64)+1, ctx.j0_int.type(th.int64)+1]
#         wy4 = (-0.5*ctx.n0) * (ctx.j0-ctx.j0_int)

#         if ctx.ndim == 3:
#             wy1 = wy1.unsqueeze(2).expand_as(Dy1)
#             wy2 = wy2.unsqueeze(2).expand_as(Dy2)
#             wy3 = wy3.unsqueeze(2).expand_as(Dy3)
#             wy4 = wy4.unsqueeze(2).expand_as(Dy4)

#         df0_dy0 = Dy1 * wy1 + Dy2 * wy2 + Dy3 * wy3 + Dy4 * wy4

#         if ctx.ndim == 3:
#             ctx.x = ctx.x.unsqueeze(2).expand_as(Dx1)
#             ctx.y = ctx.y.unsqueeze(2).expand_as(Dy1)

#         grad_s = df0_dx0 * ctx.x + df0_dy0 * ctx.y
#         # plt.imshow(grad_s.cpu().numpy(), vmin=-100, vmax=100)
#         # plt.colorbar()
#         # plt.title('Autograd: scale = %.1f' % s.item())
#         # plt.savefig('autodiff_%.1f.png' % s.item())
#         # plt.close()

#         grad_s_out = th.zeros(1, dtype=th.float32, device=ctx.I0.device)
#         grad_s_out[0] = (grad_output * grad_s).sum()
       
#         return None, grad_s_out, None