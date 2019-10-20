import numpy as np
import torch as th
from utility import *
from descriptor import *
from torchvision.transforms import Normalize, Compose

class Bins:
    def __init__(self, target, imsize, device):
        print("Initial Class sumfunc::Bins()")
        self.imsize = imsize
        self.device = device
        self.imres = target.shape[0]
        self.target = th.from_numpy(target).to(device)
        
        self.useFFTimg = True
        self.initBinsR([16,0, 8,1])
        self.binTargetMu = self.evalBins(self.target)
        if self.useFFTimg:
            self.binTargetFFTMu = self.evalBinsFFT(self.target)
        
    def initBinsR(self, nbins):
        c = self.imsize/2.0;
        unit = c/nbins[0]

        # surface positions
        v = th.arange(self.imres, dtype=th.float32, device=th.device("cpu"))
        v = ((v + 0.5) / self.imres - 0.5) * self.imsize
        y, x = th.meshgrid((v, v))
        pos = th.stack((x, -y, th.zeros_like(x)), 2)
        pos_norm = pos.norm(2.0, 2)

        _binBases = [[] for i in range(nbins[0])]
        for i in range(self.imres):
            for j in range(self.imres):
                k = int(pos_norm[i][j].item()/unit)
                if k < nbins[0]:
                    _binBases[k].append(i*self.imres+j)

        self.binBasesT = th.zeros([nbins[0]-nbins[1], self.imres*self.imres], device=th.device("cpu"));
        for i in range(nbins[1], nbins[0]):
            unit = 1.0/len(_binBases[i])
            for j in  _binBases[i]:
                self.binBasesT[i - nbins[1]][j] = unit
        
        self.binBasesT = self.binBasesT.to(self.device)

        if self.useFFTimg:
            unitFFT = c/nbins[2]
            _binFFTBases = [[] for i in range(nbins[2])]
            for i in range(self.imres):
                for j in range(self.imres):
                    k = int(pos_norm[i][j].item()/unitFFT)
                    if k < nbins[2]:
                        _binFFTBases[k].append(i*self.imres+j)

            self.binFFTBasesT = th.zeros([nbins[2]-nbins[3], self.imres*self.imres], device=th.device("cpu"));
            for i in range(nbins[3], nbins[2]):
                unitFFT = 1.0/len(_binFFTBases[i])
                for j in  _binFFTBases[i]:
                    self.binFFTBasesT[i - nbins[3]][j] = unitFFT
        
            self.binFFTBasesT = self.binFFTBasesT.to(self.device);           



    def evalBins(self, img):
        return th.mm(self.binBasesT, img.view(self.imres*self.imres, 3))

    def evalBinsFFT(self, img):
        imgFFT = img.min(2)
        imgFFT = imgFFT[0]
        imgFFT = shift(th.stack((imgFFT, th.zeros_like(imgFFT)), 2)).fft(2)
        imgFFT = shift(imgFFT.norm(2.0, 2).log1p())
        return th.mm(self.binFFTBasesT, imgFFT.view(self.imres*self.imres, 1)).view(-1)        
        
    def logpdf(self, img):
        binThisMu = self.evalBins(img)
        binSigma = 0.2 * self.binTargetMu
        lpdf = ((binThisMu - self.binTargetMu).pow(2.0)/(2.0*binSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binSigma).log()).sum()

        if self.useFFTimg:
            binThisFFTMu = self.evalBinsFFT(img)
            binFFTSigma = 0.5*self.binTargetFFTMu
            lpdf += ((binThisFFTMu - self.binTargetFFTMu).pow(2.0)/(2.0*binFFTSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binFFTSigma).log()).sum()

        return lpdf






class T_G:
    def __init__(self, target, device):
        self.device = device
        self.transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

        self.td = TextureDescriptor(device)
        # freeze the weights of td
        for p in self.td.parameters():
            p.requires_grad = False

        target = th.from_numpy(target).to(device)

        self.td_target = self.td(self.hdr2ldr(target))
        self.td_target.requires_grad = False

    def hdr2ldr(self, img):
        return self.transform(img.clamp(0,1).pow(1/2.2).permute(2,0,1))

    def logpdf(self, img):
        td_this = self.td(self.hdr2ldr(img))

        sigma = th.max(0.2 * self.td_target, th.tensor(0.01,device=self.device))
        lpdf = ((td_this - self.td_target).pow(2.0)/(2.0*sigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*sigma).log()).mean()

        return lpdf
