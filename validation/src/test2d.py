import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import multivariate_normal
from datetime import datetime
import time
import exr
epsilon = 1e-6

class Dist:
    def __init__(self, fn):
        self.loadDist(fn)
        self.comDiff()

    def loadDist(self, fn):
        self.dist = exr.read(fn)

        imgH,imgW = np.shape(self.dist)
        assert(imgH==imgW)
        self.imRes = imgH

        self.pixelUnit = 2/self.imRes
        self.lb = -1
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
        exr.write(self.diff[:,:,0], fn[:-3]+'_diffx.exr')
        exr.write(self.diff[:,:,1], fn[:-3]+'_diffy.exr')

class MCMCsample:
    def __init__(self, pdf, T):
        print("Initial Base Class MCMCsample()")
        self.pdf = pdf
        self.T = T
        self.xs = []
        self.num_accept = 0
        self.num_reject = 0

    def U(self, x):
        imRes = self.pdf.imRes
        if x[0] <= self.pdf.lb or x[1] <= self.pdf.lb or x[0] >= self.pdf.ub or x[1] >= self.pdf.ub:
            return np.inf
        j = int(np.round((x[0]+1)/2*imRes-0.5))
        i = int(np.round((x[1]+1)/2*imRes-0.5))
        return -np.log(max(epsilon, self.pdf.dist[i, j]))

    def dU(self, x):
        imRes = self.pdf.imRes
        if x[0] <= self.pdf.lb or x[1] <= self.pdf.lb or x[0] >= self.pdf.ub or x[1] >= self.pdf.ub:
            return 0
        # print(x)
        j = int(np.round((x[0]+1)/2*imRes-0.5))
        i = int(np.round((x[1]+1)/2*imRes-0.5))
        return -self.pdf.diff[i, j, :] / max(epsilon, self.pdf.dist[i, j])

    def mcmc(self):
        return 1, 0

    def sample(self, x0):
        self.xs.append(x0.copy())
        initTime = time.time()
        while (time.time() - initTime) < self.T:
            self.N = self.num_accept + self.num_reject
            if self.N%10000==0:
                print('accept:%d, reject:%d, accept rate:%.2f%%' % \
                    (self.num_accept, self.num_reject, self.num_accept*100/max(1,self.N)))

            arej, x_tmp = self.mcmc()

            if np.random.rand() < min(1, arej):
                self.num_accept += 1
                self.xs.append(x_tmp.copy())
            else:
                self.num_reject += 1

        return np.vstack(self.xs)

class HMCsample(MCMCsample):
    def __init__(self, pdf, T):
        print("Initial Class HMCsample()")
        super().__init__(pdf, T)
        self.setLeapfrogPara(0.1,5)

    def setLeapfrogPara(self, len, L):
        self.delta = len/L
        self.L = L

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

    def mcmc(self):
        p_this = np.random.randn(2)
        x_tmp, p_tmp = self.leapfrog(self.xs[-1], p_this)
        H_this = self.U(self.xs[-1]) + self.K(p_this)
        H_tmp  = self.U(x_tmp)  + self.K(p_tmp)
        arej = np.exp(H_this - H_tmp)
        return arej, x_tmp


class MALAsample(MCMCsample):
    def __init__(self, pdf, T):
        print("Initial Class MALAsample()")
        super().__init__(pdf, T)
        self.alpha = 0.9
        self.beta = 0.999
        self.c1 = 0.25
        self.c2 = 0.25
        self.delta = 0.001
        self.V1 = np.array([0,0])
        self.V2 = np.array([0,0])

    def mcmc(self):
        ###### generate proposal from current
        U_this  =  self.U(self.xs[-1])
        dU_this = self.dU(self.xs[-1])

        V1_this = self.alpha * self.V1 + (1 - self.alpha) * dU_this
        V2_this = self.beta  * self.V2 + (1 - self.beta)  * dU_this * dU_this

        M1 = max(epsilon, self.num_accept)**-self.c1 * V1_this + dU_this
        M2 = 1/(epsilon + max(epsilon, self.num_accept)**-self.c2 * np.sqrt(V2_this))

        mu = self.xs[-1] - 0.5 * self.delta * M1 * M2
        sigma2 = self.delta * M2
        sigma = np.sqrt(sigma2)
        x_tmp = mu + sigma * np.random.randn(2)
        q_this = multivariate_normal.pdf(x_tmp, mu, sigma2, allow_singular = True)

        ##### compute proposal to current
        U_tmp  =  self.U(x_tmp)
        dU_tmp = self.dU(x_tmp)

        V1_tmp = self.alpha * self.V1 + (1 - self.alpha) * dU_tmp
        V2_tmp = self.beta  * self.V2 + (1 - self.beta) *  dU_tmp * dU_tmp

        M1 = max(epsilon, self.num_accept)**-self.c1 * V1_tmp + dU_tmp
        M2 = 1/(epsilon + max(epsilon, self.num_accept)**-self.c2 * np.sqrt(V2_tmp))

        mu = x_tmp - 0.5 * self.delta * M1 * M2
        sigma2 = self.delta * M2
        q_tmp = multivariate_normal.pdf(self.xs[-1], mu, sigma2, allow_singular = True)

        ##### #######
        arej = np.exp(U_this-U_tmp) * q_tmp/q_this

        self.V1 = V1_this
        self.V2 = V2_this

        return arej, x_tmp


def KL_div(a, b):
    assert(np.shape(a) == np.shape(b))
    h,w = np.shape(a)
    kl_div = 0
    for i in range(h):
        for j in range(w):
            kl_div += a[i,j] * np.log(max(epsilon,a[i,j]) / max(epsilon,b[i,j]))
    return kl_div


def main(fn):
    in_PATH = '../in/test2d/' + fn + '/'
    out_PATH = '../out/test2d/' + fn + '/'

    pdf = Dist(in_PATH + 'pdf.exr')
    x0 = np.array((0, 0))
    T = 200

    hmc = HMCsample(pdf, T)
    print(datetime.now())
    xs_hmc = hmc.sample(x0)
    print(datetime.now())

    mala = MALAsample(pdf, T)
    print(datetime.now())
    xs_mala = mala.sample(x0)
    print(datetime.now())

    fig = plt.figure(figsize=(18,6))
    plt.subplot(131)
    plt.imshow(pdf.dist, vmin=0, vmax=max(pdf.dist.flatten()))
    plt.axis('equal')
    plt.axis('off')
    plt.title('Target pdf (%dx%d)' % (pdf.imRes, pdf.imRes))


    plt.subplot(132)
    out = np.zeros((pdf.imRes,pdf.imRes), dtype='float32')
    for k in range(hmc.num_accept):
        j = int(np.round((xs_hmc[k,0]+1)/2 * pdf.imRes - 0.5))
        i = int(np.round((xs_hmc[k,1]+1)/2 * pdf.imRes - 0.5))
        out[i,j] += 1
    out /= hmc.num_accept
    kl_div = KL_div(pdf.dist, out)
    plt.imshow(out, vmin=0, vmax=max(pdf.dist.flatten()))
    plt.axis('equal')
    plt.axis('off')
    plt.title('HMC, N=%d, acc rate=%.2f%%, KL=%.2f' % \
        (hmc.N, hmc.num_accept*100/hmc.N, kl_div))



    plt.subplot(133)
    out = np.zeros((pdf.imRes,pdf.imRes), dtype='float32')
    for k in range(mala.num_accept):
        j = int(np.round((xs_mala[k,0]+1)/2 * pdf.imRes - 0.5))
        i = int(np.round((xs_mala[k,1]+1)/2 * pdf.imRes - 0.5))
        out[i,j] += 1
    out /= mala.num_accept
    kl_div = KL_div(pdf.dist, out)
    plt.imshow(out, vmin=0, vmax=max(pdf.dist.flatten()))
    plt.axis('equal')
    plt.axis('off')
    plt.title('MALA, N=%d, acc rate=%.2f%%, KL=%.2f' % \
        (mala.N, mala.num_accept*100/mala.N, kl_div))

    plt.savefig(out_PATH + 'mcmc_sample.png')

if __name__ == '__main__':
    main('DeMes')
    main('heightmap')
    main('vortex')