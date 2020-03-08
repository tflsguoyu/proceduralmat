import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import multivariate_normal
from datetime import datetime
import time
sys.path.insert(0, '../include')
import exr
epsilon = 1e-6

class Dist:
    def __init__(self, fn):
        self.loadDist(fn)
        self.comDiff()

    def loadDist(self, fn):
        self.dist = exr.read(fn + '.exr')

        imgH,imgW = np.shape(self.dist)
        assert(imgH==imgW)
        self.imRes = imgH

        self.pixelUnit = 1/self.imRes
        self.lb = 0
        self.ub = 1

    def comDiff(self):
        self.diff = np.zeros((self.imRes,self.imRes,2), dtype=np.float32)
        for i in range(self.imRes):
            for j in range(self.imRes):
                if j > 0 and j < self.imRes-1:
                    self.diff[i,j,0] = (self.dist[i,j+1] - self.dist[i,j-1]) / (2*self.pixelUnit)
                elif j == 0:
                    self.diff[i,j,0] = (self.dist[i,j+1] - self.dist[i,j])   / (  self.pixelUnit)
                elif j == self.imRes-1:
                    self.diff[i,j,0] = (self.dist[i,j]   - self.dist[i,j-1]) / (  self.pixelUnit)

                if i > 0 and i < self.imRes-1:
                    self.diff[i,j,1] = (self.dist[i+1,j] - self.dist[i-1,j]) / (2*self.pixelUnit)
                elif i == 0:
                    self.diff[i,j,1] = (self.dist[i+1,j] - self.dist[i,j])   / (  self.pixelUnit)
                elif i == self.imRes-1:
                    self.diff[i,j,1] = (self.dist[i,j]   - self.dist[i-1,j]) / (  self.pixelUnit)

    def saveDiff(self):
        exr.write(self.diff[:,:,0], fn+'_diffx.exr')
        exr.write(self.diff[:,:,1], fn+'_diffy.exr')


class HMCsample:
    def __init__(self, pdf, N):
        self.pdf = pdf
        self.N = int(N)
        self.setLeapfrogPara(0.1,5)
        self.xs = []
        self.num_reject = 0

    def setLeapfrogPara(self, len, L):
        self.delta = len/L
        self.L = L

    def U(self, x):
        imRes = self.pdf.imRes
        if x[0] <= self.pdf.lb or x[1] <= self.pdf.lb or x[0] >= self.pdf.ub or x[1] >= self.pdf.ub:
            return np.inf
        j = int(np.round(x[0]*imRes-0.5))
        i = int(np.round(x[1]*imRes-0.5))
        return -np.log(max(epsilon, self.pdf.dist[i, j]))

    def dU(self, x):
        imRes = self.pdf.imRes
        if x[0] <= self.pdf.lb or x[1] <= self.pdf.lb or x[0] >= self.pdf.ub or x[1] >= self.pdf.ub:
            return 0
        # print(x)
        j = int(np.round(x[0]*imRes-0.5))
        i = int(np.round(x[1]*imRes-0.5))
        return -self.pdf.diff[i, j, :] / max(epsilon, self.pdf.dist[i, j])


    def K(self, p): return 0.5* np.dot(p,p)

    def dK(self, p): return p

    def leapfrog(self, x0, p0):
        p = p0 - self.delta/2 * self.dU(x0)
        x = x0 + self.delta   * self.dK(p)

        for i in range(self.L-1):
            p = p - self.delta * self.dU(x)
            x = x + self.delta * self.dK(p)

        p = p - self.delta/2 * self.dU(x)

        return x, p

    def sample(self, x0):
        # x0 = np.array((0.5, 0.5))
        self.xs.append(x0.copy())
        t = 1
        while len(self.xs) < self.N:
            if t%10000==0: print('%d/%d' % (t,self.N))
            p_this = np.random.randn(2)
            x_tmp, p_tmp = self.leapfrog(self.xs[-1], p_this)
            H_this = self.U(self.xs[-1]) + self.K(p_this)
            H_tmp  = self.U(x_tmp)  + self.K(p_tmp)
            alpha = min(1, np.exp(H_this - H_tmp))
            if np.random.rand() < alpha:
                t += 1
                self.xs.append(x_tmp.copy())
            else:
                self.num_reject += 1
                # print('reject')

        return np.vstack(self.xs)


class MALAsample:
    def __init__(self, pdf, N):
        print("Initial Class MALAsample()")
        self.pdf = pdf
        self.N = int(N)
        self.setPara(alpha=0.9, beta=0.999, c1=0.1, c2=0.1, delta=0.02)
        self.xs = []
        self.num_reject = 0

    def setPara(self, alpha, beta, c1, c2, delta):
        self.alpha = alpha
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.delta = delta

    def U(self, x):
        imRes = self.pdf.imRes
        if x[0] < self.pdf.lb or x[1] < self.pdf.lb or x[0] > self.pdf.ub or x[1] > self.pdf.ub:
            return -np.inf
        j = int(np.round(x[0]*imRes-0.5))
        i = int(np.round(x[1]*imRes-0.5))
        return np.log(max(epsilon, self.pdf.dist[i, j]))

    def g(self, x):
        imRes = self.pdf.imRes
        if x[0] < self.pdf.lb or x[1] < self.pdf.lb or x[0] > self.pdf.ub or x[1] > self.pdf.ub:
            return np.array([0,0])
        j = int(np.round(x[0]*imRes-0.5))
        i = int(np.round(x[1]*imRes-0.5))
        return self.pdf.diff[i, j, :] / max(epsilon, self.pdf.dist[i, j])

    def sample(self, x0):
        G = np.array([0,0])
        d = np.array([0,0])
        self.xs.append(x0)
        t = 1
        while len(self.xs) < self.N:
            if t%10000==0: print('%d/%d, reject rate=%.2f' % (t,self.N, self.num_reject/(self.num_reject+t)*100))


            ###### tmp
            g_this = self.g(self.xs[-1])

            G_tmp = self.beta * G + (1-self.beta) * g_this * g_this
            d_tmp = self.alpha * d + (1-self.alpha) * g_this

            M = 1/(epsilon + t**-self.c1 * np.sqrt(G_tmp))
            m = t**-self.c2 * d_tmp + g_this

            w = np.random.randn(2)
            mu = self.xs[-1] + 0.5 * self.delta * M * m
            sigma2 = self.delta * M
            sigma = np.sqrt(sigma2)
            x_tmp = mu + sigma * w

            q_this = multivariate_normal.pdf(x_tmp, mu, sigma2, allow_singular = True)

            ##### this
            g_tmp = self.g(x_tmp)

            G_this = self.beta * G + (1-self.beta) * g_tmp * g_tmp
            d_this = self.alpha * d + (1-self.alpha) * g_tmp

            M = 1/(epsilon + t**-self.c1 * np.sqrt(G_this))
            m = t**-self.c2 * d_this + g_tmp

            mu = x_tmp + 0.5 * self.delta * M * m
            sigma2 = self.delta * M

            q_tmp = multivariate_normal.pdf(self.xs[-1], mu, sigma2, allow_singular = True)

            ##### #######
            U_tmp = self.U(x_tmp)
            # print(x_tmp)
            U_this = self.U(self.xs[-1])
            # alpha = min(1, U_tmp/U_this)
            alpha = min(1, (np.exp(U_tmp) * q_tmp) / (np.exp(U_this) * q_this))
            if np.random.rand() < alpha:
                t += 1
                G = G_tmp
                d = d_tmp
                self.xs.append(x_tmp)
            else:
                self.num_reject += 1

        return np.vstack(self.xs)


def KL_div(a, b):
    assert(np.shape(a) == np.shape(b))
    h,w = np.shape(a)
    kl_div = 0
    for i in range(h):
        for j in range(w):
            kl_div += a[i,j] * np.log(max(epsilon,a[i,j]) / max(epsilon,b[i,j]))
    return kl_div


def main(flag):
    fn = 'test2d/vortex/vortex'
    pdf = Dist(fn)
    x0 = np.array((0.5, 0.5))
    xs = []

    if flag == 1:
        hmc = HMCsample(pdf, 1e6)
        timeElapse = time.time()
        print(datetime.now())
        xs = hmc.sample(x0)
        timeElapse = time.time() - timeElapse
        print(datetime.now())
    if flag == 2:
        mala = MALAsample(pdf, 1e6)
        timeElapse = time.time()
        print(datetime.now())
        xs = mala.sample(x0)
        timeElapse = time.time() - timeElapse
        print(datetime.now())

    out = np.zeros((pdf.imRes,pdf.imRes), dtype='float32')
    for k in range(xs.shape[0]):
        j = int(np.round(xs[k,0] * pdf.imRes - 0.5))
        i = int(np.round(xs[k,1] * pdf.imRes - 0.5))
        out[i,j] += 1
    out /= xs.shape[0]


    kl_div = KL_div(pdf.dist, out)

    ###
    fig = plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.imshow(pdf.dist, vmin=0, vmax=max(pdf.dist.flatten()))
    plt.axis('equal')
    plt.axis('off')
    plt.title('Target pdf (%dx%d)' % (pdf.imRes, pdf.imRes))


    plt.subplot(122)
    plt.imshow(out, vmin=0, vmax=max(pdf.dist.flatten()))
    plt.axis('equal')
    plt.axis('off')
    if flag == 1:
        plt.title('HMC, success=%d, reject=%d, time=%ds' % (xs.shape[0], hmc.num_reject, timeElapse))
        plt.savefig(fn+'_hmc.png')
    elif flag == 2:
        plt.title('MALA, success=%d, reject=%d, time=%ds' % (xs.shape[0], mala.num_reject, timeElapse))
        plt.savefig(fn+'_mala.png')
    plt.show()

if __name__ == '__main__':
    # main(0)
    # main(1)
    main(2)