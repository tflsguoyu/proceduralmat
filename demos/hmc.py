import numpy as np
import torch as th
from utility import *

class HMC:
    def __init__(self, pdf, N):
        print("Initial Class HMC()")
        self.setEpsilon(0.000001)
        self.pdf = pdf
        self.ub = pdf.ub
        self.lb = pdf.lb
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
        j = int(np.round(x[0]*self.pdf.imRes-0.5))    
        i = int(np.round(x[1]*self.pdf.imRes-0.5))    
        return -np.log(max(self.epsilon, self.pdf.dist[i, j]))

    def dU(self, x): 
        if x[0] <= 0 or x[1] <= 0 or x[0] >= 1 or x[1] >= 1:
            return 0
        # print(x)    
        j = int(np.round(x[0]*self.pdf.imRes-0.5))    
        i = int(np.round(x[1]*self.pdf.imRes-0.5))    
        return -self.pdf.diff[i, j, :] / max(self.epsilon, self.pdf.dist[i, j])

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
            if (len(self.xs)+1)%10==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
            p0 = np.random.randn(2)
            x, p = self.leapfrog(x0, p0)
            if x[0] < self.ub and x[1] < self.ub \
                and x[0] > self.lb and x[1] > self.lb:
                H0 = self.U(x0) + self.K(p0)
                H  = self.U(x)  + self.K(p)
                alpha = min(1, np.exp(H0 - H))
                if np.random.rand() < alpha:
                    self.xs.append(x.copy())
                    self.lpdfs.append(self.lpdf.detach().cpu().numpy())
                    x0 = x
                else:
                    self.num_reject += 1
            else:
                self.num_outBound += 1
        
        self.xs = np.vstack(self.xs)

    def hist(self, res):
        self.out = np.zeros((res,res), dtype='float32')
        for k in range(self.N):
            j = int(np.round(self.xs[k,0] * res - 0.5))
            i = int(np.round(self.xs[k,1] * res - 0.5))
            self.out[i,j] += 1
        self.out /= self.N

class HMCBumpTest(HMC):
    def __init__(self, target, mfbObj, N):
        print("Initial Class HMCBump()")
        self.setEpsilon(0.000001)
        self.N = N
        self.setLeapfrogPara(0.001, 20)
        self.xs = []
        self.lpdfs = []
        self.num_reject = 0
        self.num_outBound = 0
        self.lb = 0
        self.ub = 0.5

        self.mfbObj = mfbObj
        self.n = mfbObj.n
        self.initBinsR([16,0])
        self.binMu = self.evalBins(target)
        
    def initBinsR(self, nbins):
        c = self.mfbObj.size/2.0;
        unit = c/nbins[0]
        tmp = self.mfbObj.pos_norm.cpu();

        _binBases = [[] for i in range(nbins[0])]
        for i in range(self.n):
            for j in range(self.n):
                k = int(tmp[i][j].item()/unit)
                if k < nbins[0]:
                    _binBases[k].append(i*self.n+j)

        self.binBasesT = th.zeros([nbins[0]-nbins[1], self.n*self.n], device=th.device("cpu"));
        for i in range(nbins[1], nbins[0]):
            unit = 1.0/len(_binBases[i])
            for j in  _binBases[i]:
                self.binBasesT[i - nbins[1]][j] = unit
        
        self.binBasesT = self.binBasesT.to(self.mfbObj.device);        


    def evalBins(self, img):
        return th.mm(self.binBasesT, img.view(self.n*self.n, 3))

    def logpdf(self, bins, binMu):
        binSigma = 0.2*binMu
        return ((bins - binMu).pow(2.0)/(2.0*binSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binSigma).log()).sum()

    def logPrior(self, y, a, b):
        px = 1/(b-a)  # uniform distribution
        invLogitY = invLogit(y)
        return -np.log(px*(a+(b-a)*invLogitY)*(b-a)*invLogitY*(1.0-invLogitY))


        
    def U(self, x):
        para = [625, 0.42, 0.1, 0.1, x[0], 3.8, 0.02, 7]
        bins = self.evalBins(self.mfbObj.eval(para))
        self.lpdf = self.logpdf(bins, self.binMu)
        return self.lpdf.detach().cpu().numpy()
    
    def dU(self, x):
        self.U(x)
        # print(self.lpdf)
        self.lpdf.backward()
        # return [self.mfbObj.light.grad.item(), \
        #         self.mfbObj.albedo[0].grad.item(), \
        #         self.mfbObj.albedo[1].grad.item(), \
        #         self.mfbObj.albedo[2].grad.item(), \
        #         self.mfbObj.rough.grad.item(), \
        #         self.mfbObj.fsigma.grad.item(), \
        #         self.mfbObj.fscale.grad.item(), \
        #         self.mfbObj.iSigma.grad.item()]
        # return np.array([self.mfbObj.fsigma.grad.item(), \
        #                  self.mfbObj.fscale.grad.item()])
        return np.array([self.mfbObj.rough.grad.item()])        

    def sample(self, x0):
        numOfPara = len(x0)
        while len(self.xs) < self.N:
            if (len(self.xs)+1)%10==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
            p0 = np.random.randn(numOfPara)
            x, p = self.leapfrog(x0, p0)

            if x[0] > self.lb and x[0] < self.ub:
                H0 = self.U(x0) + self.K(p0)
                H  = self.U(x)  + self.K(p)
                alpha = min(1, np.exp(H0 - H))
                if np.random.rand() < alpha:
                    x0 = x
                    self.xs.append(x.copy())
                    self.lpdfs.append(self.lpdf.detach().cpu().numpy())
                else:
                    print("[reject]")
                    self.num_reject += 1
        
        self.xs = np.vstack(self.xs)        

class HMCBump(HMC):
    def __init__(self, target, mfbObj, N):
        print("Initial Class HMCBump()")
        self.setEpsilon(0.000001)
        self.N = N
        self.setLeapfrogPara(0.01, 20)
        self.xs = []
        self.lpdfs = []
        self.num_reject = 0
        self.num_outBound = 0
        self.lb = 0
        self.ub = 0.5

        self.mfbObj = mfbObj
        self.device = mfbObj.device
        self.n = mfbObj.n

        self.useFFTimg = True
        self.initBinsR([16,0, 16,0])
        self.binMu = self.evalBins(target)
        if self.useFFTimg:
            self.binFMu = self.evalBinsF(target)
        
    def initBinsR(self, nbins):
        c = self.mfbObj.size/2.0;
        unit = c/nbins[0]
        tmp = self.mfbObj.pos_norm.cpu();

        _binBases = [[] for i in range(nbins[0])]
        for i in range(self.n):
            for j in range(self.n):
                k = int(tmp[i][j].item()/unit)
                if k < nbins[0]:
                    _binBases[k].append(i*self.n+j)

        self.binBasesT = th.zeros([nbins[0]-nbins[1], self.n*self.n], device=th.device("cpu"));
        for i in range(nbins[1], nbins[0]):
            unit = 1.0/len(_binBases[i])
            for j in  _binBases[i]:
                self.binBasesT[i - nbins[1]][j] = unit
        
        self.binBasesT = self.binBasesT.to(self.device);  

        if self.useFFTimg:
            unitF = c/nbins[2]
            _binFBases = [[] for i in range(nbins[2])]
            for i in range(self.n):
                for j in range(self.n):
                    k = int(tmp[i][j].item()/unitF)
                    if k < nbins[2]:
                        _binFBases[k].append(i*self.n+j)

            self.binFBasesT = th.zeros([nbins[2]-nbins[3], self.n*self.n], device=th.device("cpu"));
            for i in range(nbins[3], nbins[2]):
                unitF = 1.0/len(_binFBases[i])
                for j in  _binFBases[i]:
                    self.binFBasesT[i - nbins[3]][j] = unitF
            
            self.binFBasesT = self.binFBasesT.to(self.device);              



    def evalBins(self, img):
        bins = th.mm(self.binBasesT, img.view(self.n*self.n, 3))
        return bins

    def evalBinsF(self, img):
        imgF = img.min(2)
        imgF = imgF[0]
        imgF = shift(th.stack((imgF, th.zeros_like(imgF)), 2)).fft(2)
        imgF = shift(imgF.norm(2.0, 2).log1p())
        binsF = th.mm(self.binFBasesT, imgF.view(self.n*self.n, 1)).view(-1)        
        return binsF


    def logpdf(self):
        binSigma = 0.2*self.binMu
        lpdf = ((self.bins - self.binMu).pow(2.0)/(2.0*binSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binSigma).log()).sum()

        if self.useFFTimg:
            binFSigma = 0.5*self.binFMu
            lpdf += ((self.binsF - self.binFMu).pow(2.0)/(2.0*binFSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binFSigma).log()).sum()

        return lpdf

    def logPrior(self, y, a, b):
        def pxUniform(x):
            return 1/(b-a)  # uniform distribution
        def pxGaussian(x):
            return 1/(np.sqrt(2*np.pi)*0.1)*np.exp(-pow(x-0.25,2)/(2*0.1*0.1))
        
        invLogitV = invLogit(y)
        return -np.log(pxUniform(a+(b-a)*invLogitV)*(b-a)*invLogitV*(1.0-invLogitV))
        # return -np.log(self.epsilon + pxGaussian(a+(b-a)*invLogitV)*(b-a)*invLogitV*(1.0-invLogitV))


        
    def U(self, x):
        para = [625, 0.42, 0.1, 0.1, invTransVar(x[0],self.lb,self.ub), 3.8, 0.02, 7]
        img = self.mfbObj.eval(para)
        
        self.bins = self.evalBins(img)
        if self.useFFTimg:
            self.binsF = self.evalBinsF(img)

        self.lpdf = self.logpdf() + \
                    self.logPrior(x[0], self.lb, self.ub)
        return self.lpdf.detach().cpu().numpy()
    
    def dU(self, x):
        self.U(x)
        # print(self.lpdf)
        self.lpdf.backward()
        # return [self.mfbObj.light.grad.item(), \
        #         self.mfbObj.albedo[0].grad.item(), \
        #         self.mfbObj.albedo[1].grad.item(), \
        #         self.mfbObj.albedo[2].grad.item(), \
        #         self.mfbObj.rough.grad.item(), \
        #         self.mfbObj.fsigma.grad.item(), \
        #         self.mfbObj.fscale.grad.item(), \
        #         self.mfbObj.iSigma.grad.item()]
        # return np.array([self.mfbObj.fsigma.grad.item(), \
        #                  self.mfbObj.fscale.grad.item()])
        return np.array([self.mfbObj.rough.grad.item() * invTransVarGrad(x[0], self.lb, self.ub)])        

    def sample(self, x0):
        numOfPara = len(x0)
        x0[0] = transVar(x0[0], self.lb, self.ub)
        while len(self.xs) < self.N:
            if (len(self.xs)+1)%10==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
            p0 = np.random.randn(numOfPara)
            x, p = self.leapfrog(x0, p0)

            H0 = self.U(x0) + self.K(p0)
            H  = self.U(x)  + self.K(p)
            alpha = min(1, np.exp(H0 - H))
            if np.random.rand() < alpha:
                x0 = x
                x[0] = invTransVar(x[0], self.lb, self.ub)
                self.xs.append(x.copy())
                self.lpdfs.append(self.lpdf.detach().cpu().numpy())
            else:
                print("[reject]")
                self.num_reject += 1
        
        self.xs = np.vstack(self.xs)        