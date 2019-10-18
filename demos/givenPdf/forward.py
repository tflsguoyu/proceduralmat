import numpy as np
import exr

class Forward():
    def __init__(self, fn):
        print('Initial Class Dist()')
        self.genDist(fn)

    def genDist(self, fn):
        self.pdf = exr.read(fn)
        imgH,imgW = np.shape(self.pdf)
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
                    self.diff[i,j,0] = (self.pdf[i,j+1] - self.pdf[i,j-1]) / (2*self.pixelUnit)
                elif j == 0:
                    self.diff[i,j,0] = (self.pdf[i,j+1] - self.pdf[i,j])   / (  self.pixelUnit)
                elif j == self.imRes-1:
                    self.diff[i,j,0] = (self.pdf[i,j]   - self.pdf[i,j-1]) / (  self.pixelUnit)
                        
                if i > 0 and i < self.imRes-1:
                    self.diff[i,j,1] = (self.pdf[i+1,j] - self.pdf[i-1,j]) / (2*self.pixelUnit)
                elif i == 0:
                    self.diff[i,j,1] = (self.pdf[i+1,j] - self.pdf[i,j])   / (  self.pixelUnit)
                elif i == self.imRes-1:
                    self.diff[i,j,1] = (self.pdf[i,j]   - self.pdf[i-1,j]) / (  self.pixelUnit)

