from util import *

def p2x(p, n):
    return (p*2 + 1) / n - 1
def x2p(x, n):
    p = ((1+x) * n - 1) / 2
    return p%n

class Material:
    def __init__(self, args, device):
        self.device = device
        self.dir = args.out_dir
        self.n = args.imres
        self.size = args.size
        self.camera = args.camera
        self.initGeometry()

    def loadPara(self, para, paraId, ifPlot=False):
        self.para = para
        self.paraId = paraId
        if ifPlot: self.plotPrior()
        # exit()

    def sample_prior(self):
        xs = []
        for p in self.paraPr:
            x = gySampleVar(p[0].item(), p[1].item(), p[2].item(), p[3].item())
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

        # print(paraList)
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
