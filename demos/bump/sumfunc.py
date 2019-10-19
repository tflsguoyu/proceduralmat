import numpy as np
import torch as th
from utility import *

from torchvision.models.vgg import vgg19_bn

class Bins:
    def __init__(self, target, imsize, device):
        print("Initial Class sumfunc::Bins()")
        self.imsize = imsize
        self.device = device
        self.imres = target.shape[0]
        self.target = th.from_numpy(target).float().to(device)
        
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






class TextureDescriptor(th.nn.Module):

    def __init__(self, target, device):
        super(TextureDescriptor, self).__init__()
        self.device = device
        self.target = th.from_numpy(target).float().to(device)

        # get VGG19 feature network in evaluation mode
        self.net = vgg19_bn(True).features.to(device).eval()

        self.outputs = []
        def hook(module, input, output):
            self.outputs.append(output)

        for i in [6, 13, 26, 39]:
            self.net[i].register_forward_hook(hook)

        # weight proportional to num. of feature channels [Aittala 2016]
        self.weights = [1, 2, 4, 8, 8]

        self.loss = th.nn.L1Loss()
        tmp = self.floatToTd(self.target)
        self.td_target = self.forward(tmp)

    def floatToTd(self, img):
        return img.pow(1/2.2).permute(2,0,1).unsqueeze(0) *255-128

    def forward(self, x):
        # run VGG features
        self.outputs = []
        x = self.net(x)
        self.outputs.append(x)

        result = []
        for i, F in enumerate(self.outputs):
            F = F.squeeze()
            f, s1, s2 = F.shape
            s = s1 * s2
            F = F.view((f, s))

            # Gram matrix
            G = th.mm(F, F.t()) / s
            result.append(G.flatten() * self.weights[i])

        return th.cat(result)

    def logpdf(self, img):
        tmp = self.floatToTd(img)
        td_this = self.forward(tmp)
        return self.loss(td_this, self.td_target)
