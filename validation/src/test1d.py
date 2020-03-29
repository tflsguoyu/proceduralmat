import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import multivariate_normal
import time
epsilon = 1e-6

class Dist:
    def __init__(self, N):
        print("Initial Class Dist()")
        self.genDist(N)
        self.comDiff()

    def genDist(self, N):
        loc = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
        scale = np.array([0.06, 0.08, 0.10, 0.12, 0.14])
        # loc *= 0.1
        # scale *= 0.1

        self.lb = -1
        self.ub = 1
        self.x = np.linspace(self.lb, self.ub, N)

        pdf = stats.norm.pdf
        self.dist = 0
        for i in range(5):
            self.dist += pdf(self.x, loc=loc[i], scale=scale[i])
        self.dist /= 5

    def comDiff(self):
        self.diff = self.dist.copy()
        for i in range(len(self.x)-2):
            self.diff[i+1] = (self.dist[i+2] - self.dist[i]) / (self.x[i+2] - self.x[i])
        self.diff[0] = self.diff[1]
        self.diff[-1] = self.diff[-2]

    def printDist(self):
        print('\n# Print Dist() info: ')
        print('## Dist() range: ')
        print(self.x)
        print('## Dist() distribution: ')
        print(self.dist)
        if self.diff != []:
            print('## Dist() gradient: ')
            print(self.diff)

class MCMCsample:
    def __init__(self, pdf, T):
        print("Initial Base Class MCMCsample()")
        self.pdf = pdf
        self.T = T
        self.xs = []
        self.num_accept = 0
        self.num_reject = 0

    def U(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return np.inf
        return -np.log(max(epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()]))

    def dU(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return 0
        return -self.pdf.diff[np.abs(self.pdf.x - x).argmin()] / \
            max(epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()])

    def mcmc(self):
        return 1, 0

    def sample(self, x0):
        self.xs.append(x0)
        initTime = time.time()
        while (time.time() - initTime) < self.T:
            self.N = self.num_accept + self.num_reject
            if self.N%10000==0:
                print('accept:%d, reject:%d, accept rate:%.2f%%' % \
                    (self.num_accept, self.num_reject, self.num_accept*100/max(1,self.N)))

            arej, x_tmp = self.mcmc()

            if np.random.rand() < min(1, arej):
                self.num_accept += 1
                self.xs.append(x_tmp)
            else:
                self.num_reject += 1

        return self.xs

class MHsample(MCMCsample):
    def __init__(self, pdf, T):
        print("Initial Class MHsample()")
        super().__init__(pdf, T)
        self.setPara(sigma = 0.05)

    def setPara(self, sigma):
        self.sigma = sigma

    def mcmc(self):
        delta = self.sigma * np.random.randn()
        x_tmp = self.xs[-1] + delta
        U_tmp = self.U(x_tmp)
        U_this = self.U(self.xs[-1])
        arej = np.exp(U_this-U_tmp)
        return arej, x_tmp

class HMCsample(MCMCsample):
    def __init__(self, pdf, T):
        print("Initial Class HMCsample()")
        super().__init__(pdf, T)
        self.setLeapfrogPara(0.25,5)

    def setLeapfrogPara(self, len, L):
        self.delta = len/L
        self.L = L

    def K(self, p): return 0.5*p*p

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
        p_this = np.random.randn()
        x_tmp, p_tmp = self.leapfrog(self.xs[-1], p_this)
        H_this = self.U(self.xs[-1]) + self.K(p_this)
        H_tmp  = self.U(x_tmp)  + self.K(p_tmp)
        arej = np.exp(H_this - H_tmp)
        return arej, x_tmp

class MALAsample(MCMCsample):
    def __init__(self, pdf, T):
        print("Initial Class MALAsample()")
        super().__init__(pdf, T)
        self.setPara(alpha=0.9, beta=0.999, c=0.25, delta=0.005)

    def setPara(self, alpha, beta, c, delta):
        self.alpha = alpha
        self.beta = beta
        self.c1 = c
        self.c2 = c
        self.delta = delta
        self.V1 = 0
        self.V2 = 0

    def mcmc(self):
        ##### tmp
        U_this  =  self.U(self.xs[-1])
        dU_this = self.dU(self.xs[-1])

        V1_tmp = self.alpha * self.V1 + (1 - self.alpha) * dU_this
        V2_tmp = self.beta  * self.V2 + (1 - self.beta)  * dU_this * dU_this

        M1 = max(epsilon, self.num_accept)**-self.c1 * V1_tmp + dU_this
        M2 = 1/(epsilon + max(epsilon, self.num_accept)**-self.c2 * np.sqrt(V2_tmp))

        mu = self.xs[-1] - 0.5 * self.delta * M1 * M2
        sigma2 = self.delta * M2
        sigma = np.sqrt(sigma2)
        x_tmp = mu + sigma * np.random.randn()
        q_this = multivariate_normal.pdf(x_tmp, mu, sigma2, allow_singular = True)

        ##### this
        U_tmp  =  self.U(x_tmp)
        dU_tmp = self.dU(x_tmp)

        V1_this = self.alpha * self.V1 + (1 - self.alpha) * dU_tmp
        V2_this = self.beta  * self.V2 + (1 - self.beta) *  dU_tmp * dU_tmp

        M1 = max(epsilon, self.num_accept)**-self.c1 * V1_this + dU_tmp
        M2 = 1/(epsilon + max(epsilon, self.num_accept)**-self.c2 * np.sqrt(V2_this))

        mu = x_tmp - 0.5 * self.delta * M1 * M2
        sigma2 = self.delta * M2
        q_tmp = multivariate_normal.pdf(self.xs[-1], mu, sigma2, allow_singular = True)

        ##### #######
        arej = np.exp(U_this-U_tmp) * q_tmp/q_this

        self.V1 = V1_tmp
        self.V2 = V2_tmp

        return arej, x_tmp
###
def main():
    out_PATH = '../out/test1d/'

    # http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S10E_HMC.html

    pdf = Dist(2000)
    x0 = 0
    T = 20

    mh = MHsample(pdf, T)
    print(datetime.now())
    xs_mh = mh.sample(x0)
    print(datetime.now())

    hmc = HMCsample(pdf, T)
    print(datetime.now())
    xs_hmc = hmc.sample(x0)
    print(datetime.now())

    mala = MALAsample(pdf, T)
    print(datetime.now())
    xs_mala = mala.sample(x0)
    print(datetime.now())


    fig = plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.plot(pdf.x, pdf.dist, label='Target pdf')
    plt.hist(xs_mh, 200, density=True, label='MH samples')
    plt.title('MH, N=%d, acc rate=%.2f%%' % (mh.N, mh.num_accept*100/mh.N))
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(pdf.x, pdf.dist, label='Target pdf')
    plt.hist(xs_hmc, 200, density=True, label='HMC samples')
    plt.title('HMC, N=%d, acc rate=%.2f%%' % (hmc.N, hmc.num_accept*100/hmc.N))
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(pdf.x, pdf.dist, label='Target pdf')
    plt.hist(xs_mala, 200, density=True, label='MALA samples')
    plt.title('MALA, N=%d, acc rate=%.2f%%' % (mala.N, mala.num_accept*100/mala.N))
    plt.legend()

    plt.savefig(out_PATH + 'mcmc_sample.png')

if __name__ == '__main__':
    main()
