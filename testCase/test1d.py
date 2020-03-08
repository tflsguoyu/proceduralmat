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
        loc = np.array([-4, -2, 0, 2, 4])
        # scale = np.array([0.4, 0.6, 0.3, 0.7, 0.5])
        scale = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        # scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        self.lb = loc[0] - scale[0] * 3
        self.ub = loc[-1] + scale[-1] * 3
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

# class MCMCsample:
#     def __init__(self, pdf, N):
#         print("Initial Class MCMCsample()")
#         self.pdf = pdf
#         self.N = N
#         self.setPara(0.05)
#         self.xs = []

#     def setPara(self, w):
#         self.w = w

#     def U(self, x):
#         if x < self.pdf.lb or x > self.pdf.ub:
#             return 0
#         return self.pdf.dist[np.abs(self.pdf.x - x).argmin()]

#     def sample(self, x0):
#         x = x0
#         while len(self.xs) < self.N:
#             if (len(self.xs)+1)%1000==0: print('%d/%d' % (len(self.xs)+1,self.N))

#             y = np.random.uniform(0, self.U(x))
#             lb = x
#             rb = x
#             while y < self.U(lb):
#                 lb -= self.w
#             while y < self.U(rb):
#                 rb += self.w
#             x = np.random.uniform(lb, rb)
#             if y > self.U(x):
#                 if np.abs(x-lb) < np.abs(x-rb):
#                     lb = x
#                 else:
#                     lb = y
#             else:
#                 self.xs.append(x)

class MHsample:
    def __init__(self, pdf, N):
        print("Initial Class MCMCsample()")
        self.pdf = pdf
        self.N = int(N)
        self.setPara(sigma = 0.2)
        self.xs = []
        self.num_reject = 0

    def setPara(self, sigma, T=1):
        self.sigma = sigma
        self.T = T

    def U(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return 0
        return self.pdf.dist[np.abs(self.pdf.x - x).argmin()]

    def sample(self, x0):
        self.xs.append(x0)
        t = 1
        while len(self.xs) < self.N:
            if t%10000==0: print('%d/%d' % (t,self.N))

            delta = np.random.normal(0, self.sigma)
            x_tmp = self.xs[-1] + delta
            alpha = min(1, self.U(x_tmp)/self.U(self.xs[-1]))
            if np.random.rand() < alpha:
                t += 1
                self.xs.append(x_tmp)
            else:
                self.num_reject += 1

        return self.xs

class HMCsample:
    def __init__(self, pdf, N):
        print("Initial Class HMCsample()")
        self.pdf = pdf
        self.N = int(N)
        self.setLeapfrogPara(0.4,2)
        self.xs = []
        self.num_reject = 0

    def setLeapfrogPara(self, len, L):
        self.delta = len/L
        self.L = L

    def U(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return np.inf
        return -np.log(max(epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()]))

    def dU(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return 0
        return -self.pdf.diff[np.abs(self.pdf.x - x).argmin()] / \
            max(epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()])

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

    def sample(self, x0):
        self.xs.append(x0)
        t = 1
        while len(self.xs) < self.N:
            if t%10000==0: print('%d/%d' % (t,self.N))

            p_this = np.random.randn()
            x_tmp, p_tmp = self.leapfrog(self.xs[-1], p_this)
            H_this = self.U(self.xs[-1]) + self.K(p_this)
            H_tmp  = self.U(x_tmp)  + self.K(p_tmp)
            alpha = min(1, np.exp(H_this - H_tmp))
            if np.random.rand() < alpha:
                t += 1
                self.xs.append(x_tmp)
            else:
                self.num_reject += 1

        return self.xs

class MALAsample:
    def __init__(self, pdf, N):
        print("Initial Class MALAsample()")
        self.pdf = pdf
        self.N = int(N)
        self.setPara(alpha=0.9, beta=0.999, c1=0.3, c2=0.3, delta=0.01)
        self.xs = []
        self.num_reject = 0

    def setPara(self, alpha, beta, c1, c2, delta):
        self.alpha = alpha
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.delta = delta

    def U(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return np.inf
        return -np.log(max(epsilon ,self.pdf.dist[np.abs(self.pdf.x - x).argmin()]))

    def g(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return 0
        return -self.pdf.diff[np.abs(self.pdf.x - x).argmin()] / \
            max(epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()])

    def sample(self, x0):
        G = 0
        d = 0
        self.xs.append(x0)
        t = 1
        while len(self.xs) < self.N:
            if t%1000==0: print('%d/%d, reject rate=%.2f' % (t,self.N, self.num_reject/(self.num_reject+t)*100))

            ##### tmp
            g_this = self.g(self.xs[-1])

            G_tmp = self.beta * G + (1-self.beta) * g_this * g_this
            d_tmp = self.alpha * d + (1-self.alpha) * g_this

            M = 1/(epsilon + t**-self.c1 * np.sqrt(G_tmp))
            m = t**-self.c2 * d_tmp + g_this

            mu = self.xs[-1] - 0.5 * self.delta * M * m
            sigma2 = self.delta * M
            sigma = np.sqrt(sigma2)
            x_tmp = mu - sigma * np.random.randn()

            q_this = multivariate_normal.pdf(x_tmp, mu, sigma2, allow_singular = True)

            ##### this
            g_tmp = self.g(x_tmp)

            G_this = self.beta * G + (1-self.beta) * g_tmp * g_tmp
            d_this = self.alpha * d + (1-self.alpha) * g_tmp

            M = 1/(epsilon+ t**-self.c1 * np.sqrt(G_this))
            m = t**-self.c2 * d_this + g_tmp

            mu = x_tmp - 0.5 * self.delta * M * m
            sigma2 = self.delta * M

            q_tmp = multivariate_normal.pdf(self.xs[-1], mu, sigma2, allow_singular = True)

            ##### #######
            U_tmp = self.U(x_tmp)
            U_this = self.U(self.xs[-1])
            # alpha = min(1, U_tmp/U_this)
            alpha = min(1, (np.exp(-U_tmp) * q_tmp) / (np.exp(-U_this) * q_this))
            if np.random.rand() < alpha:
                t += 1
                G = G_tmp
                d = d_tmp
                self.xs.append(x_tmp)
            else:
                self.num_reject += 1

        return self.xs

###
def main(flag):

    # flag = 0 # 0.MH, 1.HMC
    # http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S10E_HMC.html

    pdf = Dist(2000)
    x0 = 4
    xs = []

    if flag == 0:
        mh = MHsample(pdf, 1e6)
        timeElapse = time.time()
        print(datetime.now())
        xs = mh.sample(x0)
        timeElapse = time.time() - timeElapse
        print(datetime.now())
    elif flag == 1:
        hmc = HMCsample(pdf, 1e5)
        timeElapse = time.time()
        print(datetime.now())
        xs = hmc.sample(x0)
        timeElapse = time.time() - timeElapse
        print(datetime.now())
    elif flag == 2:
        mala = MALAsample(pdf, 1e5)
        timeElapse = time.time()
        print(datetime.now())
        xs = mala.sample(x0)
        timeElapse = time.time() - timeElapse
        print(datetime.now())

    # pdf.printDist()


    fig = plt.figure()
    plt.plot(pdf.x, pdf.dist, label='Target pdf')
    if flag == 0:
        plt.hist(xs, 200, density=True, label='MH samples')
        plt.title('1D MH, success=%d, reject=%d, time=%ds' % (len(xs), mh.num_reject, timeElapse))
    elif flag == 1:
        # plt.plot(pdf.x, pdf.diff, 'r', label='Target d_pdf')
        plt.hist(xs, 200, density=True, label='HMC samples')
        plt.title('1D HMC, success=%d, reject=%d, time=%ds' % (len(xs), hmc.num_reject, timeElapse))
    elif flag == 2:
        # plt.plot(pdf.x, pdf.diff, 'r', label='Target d_pdf')
        plt.hist(xs, 200, density=True, label='MALA samples')
        plt.title('1D MALA, success=%d, reject=%d, time=%ds' % (len(xs), mala.num_reject, timeElapse))
    plt.legend()
    if flag == 0: plt.savefig('test1d/mh.png')
    elif flag == 1: plt.savefig('test1d/hmc.png')
    elif flag == 2: plt.savefig('test1d/mala.png')
    # plt.show()
    plt.close()

if __name__ == '__main__':
    # main(0)
    # main(1)
    main(2)
