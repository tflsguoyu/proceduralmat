import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../include')
import exr
import hmc

class Dist():
    def __init__(self, fn):
        print('Initial Class Dist()')
        self.genDist(fn)

    def genDist(self, fn):
        self.dist = exr.read(fn + '.exr')
        imgH,imgW = np.shape(self.dist)
        assert(imgH==imgW)
        self.imRes = imgH
        self.imSize = 1
        self.pixelUnit = self.imSize/self.imRes
        self.lb = 0
        self.ub = self.imSize

    def comDiff(self):
        self.diff = np.zeros((self.imRes,self.imRes,2), dtype='float32')
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

def KL_div(a, b):
    assert(np.shape(a) == np.shape(b))
    h,w = np.shape(a)
    kl_div = 0
    epsilon = 0.000001
    for i in range(h):
        for j in range(w):
            kl_div += a[i,j] * np.log(max(epsilon,a[i,j]) / max(epsilon,b[i,j]))
    return kl_div


def main():

    fn = '../testCase/test2d/vortex/vortex'
    pdfObj = Dist(fn)
    pdfObj.comDiff()
    hmcObj = hmc.HMC(pdfObj, 100000)
    x0 = np.array([0.5,0.5])
    hmcObj.sample(x0)
    xs = hmcObj.xs
    hmcObj.hist(pdfObj.imRes)

    kl_div = KL_div(pdfObj.dist, hmcObj.out)

    # ###
    fig = plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.imshow(pdfObj.dist, vmin=0, vmax=max(pdfObj.dist.flatten()))
    plt.axis('equal')
    plt.axis('off')
    plt.title('Target pdf (%dx%d)' % (pdfObj.imRes, pdfObj.imRes))

    plt.subplot(122)
    plt.imshow(hmcObj.out, vmin=0, vmax=max(pdfObj.dist.flatten())) 
    plt.axis('equal')
    plt.axis('off')
    plt.title('HMC (%dk|%.2fk,%.2fk|%.2f)' % \
        (hmcObj.N/1000, hmcObj.num_reject/1000, hmcObj.num_outBound/1000, kl_div))

    plt.savefig(fn+'%d.png' % hmcObj.N)
    print('DONE!!!')
    plt.show()


if __name__ == '__main__':
    main()