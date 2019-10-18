import numpy as np
import torch as th

class HMC:
    def __init__(self, forward, N):
        print("Initial Class HMC()")
        self.setEpsilon(0.000001)
        self.forward = forward
        self.ub = forward.ub
        self.lb = forward.lb
        self.N = N
        self.setLeapfrogPara(0.01, 5)
        self.xs = []
        self.lpdfs = []
        self.num_reject = 0
        self.num_outBound = 0

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
 
    def U(self, x): 
        if x[0] <= 0 or x[1] <= 0 or x[0] >= 1 or x[1] >= 1:
            return 0
        j = int(np.round(x[0]*self.forward.imRes-0.5))    
        i = int(np.round(x[1]*self.forward.imRes-0.5))   
        self.lpdf = -np.log(max(self.epsilon, self.forward.pdf[i, j]))
        return self.lpdf

    def dU(self, x): 
        if x[0] <= 0 or x[1] <= 0 or x[0] >= 1 or x[1] >= 1:
            return 0
        # print(x)    
        j = int(np.round(x[0]*self.forward.imRes-0.5))    
        i = int(np.round(x[1]*self.forward.imRes-0.5))    
        return -self.forward.diff[i, j, :] / max(self.epsilon, self.forward.pdf[i, j])

    def K(self, p): return 0.5* np.dot(p,p)

    def dK(self, p): return p

    def setLeapfrogPara(self, delta, L):
        self.delta = delta
        self.L = L

    def leapfrog(self, x0, p0): 
        # print("x0:", x0)
        # print("dU(x0):", self.dU(x0))
        # print("p0:", p0)
        p = p0 - self.delta/2 * self.dU(x0)
        # print("p:", p)
        x = x0 + self.delta   * self.dK(p)
        # print("x:", x)
        for i in range(self.L-1):
            # print("x%d:" % (i+1), x)
            # print("dU(x%d):" % (i+1), self.dU(x))
            p = p - self.delta * self.dU(x)
            x = x + self.delta * self.dK(p)       
        p = p - self.delta/2 * self.dU(x)        
        return x, p

    def sample(self, x0):
        while len(self.xs) < self.N:
            if (len(self.xs)+1)%10000==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
            p0 = np.random.randn(2)
            x, p = self.leapfrog(x0, p0)
            if x[0] < self.ub and x[1] < self.ub \
                and x[0] > self.lb and x[1] > self.lb:
                H0 = self.U(x0) + self.K(p0)
                H  = self.U(x)  + self.K(p)
                alpha = min(1, np.exp(H0 - H))
                if np.random.rand() < alpha:
                    self.xs.append(x.copy())
                    self.lpdfs.append(self.lpdf)
                    x0 = x
                else:
                    self.num_reject += 1
            else:
                self.num_outBound += 1
        
        self.xs = np.vstack(self.xs)



