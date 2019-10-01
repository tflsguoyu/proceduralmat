import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

class Dist:
    def __init__(self, N):
        print("Initial Class Dist()")
        self.genDist(N)
        self.diff = []

    def genDist(self, N):
        loc1, scale1, size1 = (-2, 1, 100)
        loc2, scale2, size2 = (0, 0.3, 150)
        loc3, scale3, size3 = (2, 0.5, 100)
        x = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1), \
                            np.random.normal(loc=loc2, scale=scale2, size=size2), \
                            np.random.normal(loc=loc3, scale=scale3, size=size3), \
                             ])
        self.ub = x.max() + 1
        self.lb = x.min() - 1
        self.x = np.linspace(self.lb, self.ub, N)

        pdf = stats.norm.pdf
        self.dist = pdf(self.x, loc=loc1, scale=scale1) * float(size1) / x.size + \
                    pdf(self.x, loc=loc2, scale=scale2) * float(size2) / x.size + \
                    pdf(self.x, loc=loc3, scale=scale3) * float(size3) / x.size

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
    def __init__(self, pdf, N):
        print("Initial Class MCMCsample()")
        self.pdf = pdf
        self.N = N
        self.setPara(0.05)
        self.xs = []
    
    def setPara(self, w):
        self.w = w

    def U(self, x):
        if x < self.pdf.lb or x > self.pdf.ub:
            return 0
        return self.pdf.dist[np.abs(self.pdf.x - x).argmin()]

    def sample(self, x0):
        x = x0
        while len(self.xs) < self.N:
            if (len(self.xs)+1)%1000==0: print('%d/%d' % (len(self.xs)+1,self.N))

            y = np.random.uniform(0, self.U(x))
            lb = x
            rb = x
            while y < self.U(lb):
                lb -= self.w
            while y < self.U(rb):
                rb += self.w
            x = np.random.uniform(lb, rb)
            if y > self.U(x):
                if np.abs(x-lb) < np.abs(x-rb):
                    lb = x
                else:
                    lb = y
            else:
                self.xs.append(x)

class HMCsample:
    def __init__(self, pdf, N):
        print("Initial Class HMCsample()")
        self.setEpsilon(0.000001)
        self.pdf = pdf
        self.N = N
        self.setLeapfrogPara(0.5, 5)
        self.xs = []
        self.num_reject = 0

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
 
    def U(self, x): 
        return -np.log(max(self.epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()]))

    def dU(self, x): 
        if x < self.pdf.lb or x > self.pdf.ub:
            return 0
        return -self.pdf.diff[np.abs(self.pdf.x - x).argmin()] / \
            max(self.epsilon, self.pdf.dist[np.abs(self.pdf.x - x).argmin()])

    def K(self, p): return 0.5*p*p

    def dK(self, p): return p

    def setLeapfrogPara(self, delta, L):
        self.delta = delta
        self.L = L

    def leapfrog(self, x0, p0): 
        p = p0 - self.delta/2 * self.dU(x0)
        x = x0 + self.delta   * self.dK(p)
        for i in range(self.L-1):
            p = p - self.delta * self.dU(x)
            x = x + self.delta * self.dK(p)       
        p = p - self.delta/2 * self.dU(x)        
        return x, p

    def sample(self, x0):
        while len(self.xs) < self.N:
            if (len(self.xs)+1)%1000==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
            p0 = np.random.randn()
            x, p = self.leapfrog(x0, p0)
            if x < self.pdf.ub and x > self.pdf.lb:
                H0 = self.U(x0) + self.K(p0)
                H  = self.U(x)  + self.K(p)
                alpha = min(1, np.exp(H0 - H))
                if np.random.rand() < alpha:
                    self.xs.append(x)
                    x0 = x
                else:
                    self.num_reject += 1
            else:
                self.num_reject += 1

###
def main():

    flag = 1 # 0.MCMC, 1.HMC
    # http://people.duke.edu/~ccc14/sta-663-2018/notebooks/S10E_HMC.html

    pdf = Dist(2000)
    x0 = -1
    xs = []

    if flag == 1:
        pdf.comDiff()
        hmc = HMCsample(pdf, 100000)
        hmc.sample(x0) 
        xs = hmc.xs   
    else:
        mcmc = MCMCsample(pdf, 100000)
        mcmc.sample(x0) 
        xs = mcmc.xs   


    fig = plt.figure()
    plt.plot(pdf.x, pdf.dist, label='Target pdf')
    if flag == 1: 
        # plt.plot(pdf.x, pdf.diff, 'r', label='Target d_pdf')
        plt.hist(xs, 100, density=True, label='HMC samples')
        plt.title('1D HMC, success=%dk, reject=%.2fk' % (hmc.N/1000, hmc.num_reject/1000))
    else:
        plt.hist(xs, 100, density=True, label='MCMC samples')
        plt.title('1D MCMC, success=%dk' % (mcmc.N/1000))        
    plt.legend()
    if flag == 1: plt.savefig('test1d_hmc.png')
    else: plt.savefig('test1d_mcmc.png')
    plt.show()

if __name__ == '__main__':
    main()
