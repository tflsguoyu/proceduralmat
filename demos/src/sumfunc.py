from util import *
from descriptor import *

class Grids:
    def __init__(self, target, err_sigma, device):
        print("Initial Class sumfunc::Grids()")
        self.device = device
        self.err_sigma = err_sigma
        self.imres = target.shape[0]

        self.target = th.from_numpy(target).to(device)

        self.useFFTimg = True
        self.initBins([64,0, 16,0])
        self.binTargetMu = self.evalBins(self.target)
        if self.useFFTimg:
            self.binTargetFFTMu = self.evalBinsFFT(self.target)

    def initBins(self, nbins):
        self.nbin_mean = nbins[0]
        self.nbin_var = nbins[2]

    def evalBins(self, img):
        tmp = img.transpose(0,1).reshape((self.nbin_mean, int(self.imres*self.imres/self.nbin_mean), 3)).mean(1)
        return tmp

    def evalBinsFFT(self, img):
        imgFFT = img.min(2)
        imgFFT = imgFFT[0]
        imgFFT = gyShift(th.stack((imgFFT, th.zeros_like(imgFFT)), 2)).fft(2)
        imgFFT = gyShift(imgFFT.norm(2.0, 2).log1p())
        tmp = imgFFT.transpose(0,1).reshape((self.nbin_var, int(self.imres*self.imres/self.nbin_var))).mean(1)
        return tmp

    def logpdf(self, img):
        binThisMu = self.evalBins(img)
        binSigma = th.tensor(0.1, device=self.device)
        lpdf = ((binThisMu - self.binTargetMu).pow(2.0)/(2.0*binSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binSigma).log()).sum()

        if self.useFFTimg:
            binThisFFTMu = self.evalBinsFFT(img)
            binFFTSigma = th.tensor(0.2, device=self.device)
            lpdf += ((binThisFFTMu - self.binTargetFFTMu).pow(2.0)/(2.0*binFFTSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binFFTSigma).log()).sum()

        return lpdf

class Bins:
    def __init__(self, target, imsize, err_sigma, useFFTimg, device):
        print("Initial Class sumfunc::Bins()")
        self.imsize = imsize
        self.device = device
        self.err_sigma = err_sigma
        self.imres = target.shape[0]

        self.target = th.from_numpy(target).to(device)

        self.useFFTimg = useFFTimg
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
        imgFFT = gyShift(th.stack((imgFFT, th.zeros_like(imgFFT)), 2)).fft(2)
        imgFFT = gyShift(imgFFT.norm(2.0, 2).log1p())
        return th.mm(self.binFFTBasesT, imgFFT.view(self.imres*self.imres, 1)).view(-1)

    def logpdf(self, img):
        binThisMu = self.evalBins(img)
        binSigma = th.tensor(0.02, device=self.device)
        # binSigma = th.max(self.err_sigma[0] * self.binTargetMu,
        #     self.err_sigma[1] * th.ones_like(self.binTargetMu))

        # lpdf = ((binThisMu - self.binTargetMu).pow(2.0)/(2.0*binSigma.pow(2.0))).sum()

        lpdf = ((binThisMu - self.binTargetMu).pow(2.0)/(2.0*binSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binSigma).log()).mean()

        if self.useFFTimg:
            binThisFFTMu = self.evalBinsFFT(img)
            binFFTSigma = th.tensor(0.5, device=self.device)
            # binFFTSigma = th.max(self.err_sigma[2] * self.binTargetFFTMu,
            #     self.err_sigma[3] * th.ones_like(self.binTargetFFTMu))
            # lpdf += ((binThisFFTMu - self.binTargetFFTMu).pow(2.0)/(2.0*binFFTSigma.pow(2.0))).sum()

            lpdf += ((binThisFFTMu - self.binTargetFFTMu).pow(2.0)/(2.0*binFFTSigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*binFFTSigma).log()).mean()

        return lpdf



class T_G:
    def __init__(self, target, err_sigma, device):
        print("Initial Class sumfunc::T_G()")
        self.device = device
        self.imres = target.shape[0]
        self.err_sigma = err_sigma

        self.target = th.from_numpy(target).to(device)
        # print(self.target.type())
        # exit()

        self.td = TextureDescriptor(device)
        # freeze the weights of td
        for p in self.td.parameters():
            p.requires_grad = False

        self.cropL = int(self.imres/100*30)
        self.cropR = int(self.imres/100*70)

        self.target_R = self.target[self.cropL:self.cropR, self.cropL:self.cropR, 0].mean()
        self.target_G = self.target[self.cropL:self.cropR, self.cropL:self.cropR, 1].mean()
        self.target_B = self.target[self.cropL:self.cropR, self.cropL:self.cropR, 2].mean()
        self.targetB_R = self.target[:self.cropL, :, 0].mean()
        self.targetB_G = self.target[:self.cropL, :, 1].mean()
        self.targetB_B = self.target[:self.cropL, :, 2].mean()

        # self.target_var = self.target.reshape(self.imres*self.imres, 3).var(0)
        # print('target mean var:', self.target_mean, self.target_var)

        self.td_target = self.td(self.normalize_vgg19(self.target.permute(2,0,1)))
        # print('td_target:', self.td_target)
        self.td_target.requires_grad = False

        self.criterion = th.nn.MSELoss().to(device)

    def errloss(self, img):
        td_this = self.td(self.normalize_vgg19(img.permute(2,0,1)))
        return self.criterion(td_this, self.td_target)

    def normalize_vgg19(self, input):
        transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
        return transform(input)

    def inv_normalize_vgg19(self, input):
        transform = torchvision.transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        return transform(input)

    def logpdf(self, img):
        td_this = self.td(self.normalize_vgg19(img.permute(2,0,1)))
        sigma = th.tensor(0.01, device=self.device)
        lpdf1 = ((td_this - self.td_target).pow(2.0)/(2.0*sigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*sigma).log()).mean()

        img_R = img[self.cropL:self.cropR, self.cropL:self.cropR, 0].mean()
        img_G = img[self.cropL:self.cropR, self.cropL:self.cropR, 1].mean()
        img_B = img[self.cropL:self.cropR, self.cropL:self.cropR, 2].mean()
        imgB_R = img[:self.cropL, :, 0].mean()
        imgB_G = img[:self.cropL, :, 1].mean()
        imgB_B = img[:self.cropL, :, 2].mean()
        # print(img_R, img_G, img_B)
        sigma_mean = th.tensor(0.02, device=self.device)
        lpdf2 = (img_R - self.target_R).pow(2.0)/(2.0*sigma_mean**2.0) + \
                (img_G - self.target_G).pow(2.0)/(2.0*sigma_mean**2.0) + \
                (img_B - self.target_B).pow(2.0)/(2.0*sigma_mean**2.0) + \
                (imgB_R - self.targetB_R).pow(2.0)/(2.0*sigma_mean**2.0) + \
                (imgB_G - self.targetB_G).pow(2.0)/(2.0*sigma_mean**2.0) + \
                (imgB_B - self.targetB_B).pow(2.0)/(2.0*sigma_mean**2.0)

        lpdf = lpdf1 + lpdf2

        # print('log pdf :', lpdf1.item())
        # print('log mean:', lpdf2.item())

        # self.img_var = img.reshape(self.imres*self.imres, 3).var(0)
        # sigma_var = self.err_sigma[3]
        # lpdf += ((self.img_var - self.target_var).pow(2.0)/(2.0*sigma_var**2.0)).sum()

        return lpdf


class Mean_Var:
    def __init__(self, target, err_sigma, device):
        print("Initial Class sumfunc::Mean_Var()")
        self.device = device
        self.target = th.from_numpy(target).to(device)
        self.n = target.shape[0]

        self.target_mean = self.target.reshape(self.n*self.n, 3).mean(0)
        self.target_var = self.target.reshape(self.n*self.n, 3).var(0)
        # print('target mean var:', self.target_mean, self.target_var)

        self.loss = th.nn.MSELoss()


    def logpdf(self, img):

        sigma = th.tensor(0.02, device=self.device)
        lpdf = ((img - self.target).pow(2.0)/(2.0*sigma.pow(2.0)) + \
                (np.sqrt(2*np.pi)*sigma).log()).mean()

        # lpdf = self.loss(self.target, img)

        # sigma = th.tensor(0.01, device=self.device)
        # lpdf = ((td_this - self.td_target).pow(2.0)/(2.0*sigma.pow(2.0)) + \
        #         (np.sqrt(2*np.pi)*sigma).log()).mean()

        # self.img_mean = img.reshape(self.n*self.n, 3).mean(0)
        # sigma_mean = th.max(0.1 * self.target_mean, th.tensor(0.01,device=self.device))
        # lpdf += ((self.img_mean - self.target_mean).pow(2.0)/(2.0*sigma_mean.pow(2.0)) + \
        #         (np.sqrt(2*np.pi)*sigma_mean).log()).sum()

        # self.img_var = img.reshape(self.n*self.n, 3).var(0)
        # sigma_var = th.max(0.1 * self.target_var, th.tensor(0.001,device=self.device))
        # lpdf += ((self.img_var - self.target_var).pow(2.0)/(2.0*sigma_var.pow(2.0)) + \
        #         (np.sqrt(2*np.pi)*sigma_var).log()).sum()

        return lpdf


