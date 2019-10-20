import numpy as np
import torch as th
from utility import *

class HMC:
    def __init__(self, forwardObj, sumfuncObj, N):
        print("Initial Class hmc::HMC()")
        self.epsilon = 0.000001
        self.N = N
        self.setLeapfrogPara(0.005, 8)
        self.xs = []
        self.lpdfs = []
        self.num_reject = 0
        # self.num_outBound = 0

        self.forwardObj = forwardObj
        self.sumfuncObj = sumfuncObj
        
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
        
    def U(self, x):
        for k, id in enumerate(self.paraIdx):
            self.para[id] = x[k]
        img = self.forwardObj.eval(self.para)
        self.lpdf = self.sumfuncObj.logpdf(img)
        return self.lpdf.detach().cpu().numpy()
    
    def dU(self, x):
        self.U(x)
        self.lpdf.backward()

        U_grad = np.zeros(len(self.paraIdx))
        for k, idx in enumerate(self.paraIdx):
            U_grad[k] = self.forwardObj.para.grad[idx].item()   
        return U_grad

    def K(self, p): return 0.5* np.dot(p,p)

    def dK(self, p): return p

    def sample(self, para, paraIdx):
        self.para = para 
        self.paraIdx = paraIdx

        x0 = np.zeros(len(paraIdx))
        for k, id in enumerate(self.paraIdx):
            x0[k] = self.para[id]
        
        self.xs.append(x0.copy())
        self.lpdfs.append(9999)
        while len(self.xs) < self.N:
            if (len(self.xs)+1)%10==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
            p0 = np.random.randn(x0.size)
            x, p = self.leapfrog(x0, p0)

            H0 = self.U(x0) + self.K(p0)
            H  = self.U(x)  + self.K(p)
            alpha = min(1, np.exp(H0 - H))
            if np.random.rand() < alpha:
                x0 = x
                self.xs.append(x.copy())
                self.lpdfs.append(self.lpdf.detach().cpu().numpy())
            else:
                self.num_reject += 1
        
        self.xs = np.vstack(self.xs)
        self.lpdfs = np.vstack(self.lpdfs)        

# class HMCBump(HMC):
#     def __init__(self, target, forwardObj, N):
#         print("Initial Class HMCBump()")
#         self.setEpsilon(0.000001)
#         self.N = N
#         self.setLeapfrogPara(0.01, 20)
#         self.xs = []
#         self.lpdfs = []
#         self.num_reject = 0
#         self.num_outBound = 0
#         self.lb = 0
#         self.ub = 0.5

#         self.forwardObj = forwardObj
#         self.device = forwardObj.device
#         self.n = forwardObj.n

#         self.useFFTimg = False
#         self.initBinsR([16,0, 16,0])
#         self.binMu = self.evalBins(target)
#         if self.useFFTimg:
#             self.binFMu = self.evalBinsF(target)
        
#     def initBinsR(self, nbins):
#         c = self.forwardObj.size/2.0;
#         unit = c/nbins[0]
#         tmp = self.forwardObj.pos_norm.cpu();

#         _binBases = [[] for i in range(nbins[0])]
#         for i in range(self.n):
#             for j in range(self.n):
#                 k = int(tmp[i][j].item()/unit)
#                 if k < nbins[0]:
#                     _binBases[k].append(i*self.n+j)

#         self.binBasesT = th.zeros([nbins[0]-nbins[1], self.n*self.n], device=th.device("cpu"));
#         for i in range(nbins[1], nbins[0]):
#             unit = 1.0/len(_binBases[i])
#             for j in  _binBases[i]:
#                 self.binBasesT[i - nbins[1]][j] = unit
        
#         self.binBasesT = self.binBasesT.to(self.device);  

#         if self.useFFTimg:
#             unitF = c/nbins[2]
#             _binFBases = [[] for i in range(nbins[2])]
#             for i in range(self.n):
#                 for j in range(self.n):
#                     k = int(tmp[i][j].item()/unitF)
#                     if k < nbins[2]:
#                         _binFBases[k].append(i*self.n+j)

#             self.binFBasesT = th.zeros([nbins[2]-nbins[3], self.n*self.n], device=th.device("cpu"));
#             for i in range(nbins[3], nbins[2]):
#                 unitF = 1.0/len(_binFBases[i])
#                 for j in  _binFBases[i]:
#                     self.binFBasesT[i - nbins[3]][j] = unitF
            
#             self.binFBasesT = self.binFBasesT.to(self.device);              



#     def evalBins(self, img):
#         bins = th.mm(self.binBasesT, img.view(self.n*self.n, 3))
#         return bins

#     def evalBinsF(self, img):
#         imgF = img.min(2)
#         imgF = imgF[0]
#         imgF = shift(th.stack((imgF, th.zeros_like(imgF)), 2)).fft(2)
#         imgF = shift(imgF.norm(2.0, 2).log1p())
#         binsF = th.mm(self.binFBasesT, imgF.view(self.n*self.n, 1)).view(-1)        
#         return binsF


#     def logpdf(self):
#         binSigma = 0.2*self.binMu
#         lpdf = ((self.bins - self.binMu).pow(2.0)/(2.0*binSigma.pow(2.0)) + \
#                 (np.sqrt(2*np.pi)*binSigma).log()).sum()

#         if self.useFFTimg:
#             binFSigma = 0.5*self.binFMu
#             lpdf += ((self.binsF - self.binFMu).pow(2.0)/(2.0*binFSigma.pow(2.0)) + \
#                 (np.sqrt(2*np.pi)*binFSigma).log()).sum()

#         return lpdf

#     def logPrior(self, y, a, b):
#         def pxUniform(x):
#             return 1/(b-a)  # uniform distribution
#         def pxGaussian(x):
#             return 1/(np.sqrt(2*np.pi)*0.1)*np.exp(-pow(x-0.25,2)/(2*0.1*0.1))
        
#         invLogitV = invLogit(y)
#         return -np.log(pxUniform(a+(b-a)*invLogitV)*(b-a)*invLogitV*(1.0-invLogitV))
#         # return -np.log(self.epsilon + pxGaussian(a+(b-a)*invLogitV)*(b-a)*invLogitV*(1.0-invLogitV))


        
#     def U(self, x):
#         para = [625, 0.42, 0.1, 0.1, invTransVar(x[0],self.lb,self.ub), 3.8, 0.02, 7]
#         img = self.forwardObj.eval(para)
        
#         self.bins = self.evalBins(img)
#         if self.useFFTimg:
#             self.binsF = self.evalBinsF(img)

#         self.lpdf = self.logpdf() + \
#                     self.logPrior(x[0], self.lb, self.ub)
#         return self.lpdf.detach().cpu().numpy()
    
#     def dU(self, x):
#         self.U(x)
#         # print(self.lpdf)
#         self.lpdf.backward()
#         # return [self.forwardObj.light.grad.item(), \
#         #         self.forwardObj.albedo[0].grad.item(), \
#         #         self.forwardObj.albedo[1].grad.item(), \
#         #         self.forwardObj.albedo[2].grad.item(), \
#         #         self.forwardObj.rough.grad.item(), \
#         #         self.forwardObj.fsigma.grad.item(), \
#         #         self.forwardObj.fscale.grad.item(), \
#         #         self.forwardObj.iSigma.grad.item()]
#         # return np.array([self.forwardObj.fsigma.grad.item(), \
#         #                  self.forwardObj.fscale.grad.item()])
#         return np.array([self.forwardObj.rough.grad.item() * invTransVarGrad(x[0], self.lb, self.ub)])        

#     def sample(self, x0):
#         numOfPara = len(x0)
#         x0[0] = transVar(x0[0], self.lb, self.ub)
#         while len(self.xs) < self.N:
#             if (len(self.xs)+1)%10==0: print('%d/%d' % (len(self.xs)+1,self.N))
            
#             p0 = np.random.randn(numOfPara)
#             x, p = self.leapfrog(x0, p0)

#             H0 = self.U(x0) + self.K(p0)
#             H  = self.U(x)  + self.K(p)
#             alpha = min(1, np.exp(H0 - H))
#             if np.random.rand() < alpha:
#                 x0 = x
#                 x[0] = invTransVar(x[0], self.lb, self.ub)
#                 self.xs.append(x.copy())
#                 self.lpdfs.append(self.lpdf.detach().cpu().numpy())
#             else:
#                 print("[reject]")
#                 self.num_reject += 1
        
#         self.xs = np.vstack(self.xs)        