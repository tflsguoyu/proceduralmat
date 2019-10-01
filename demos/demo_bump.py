import numpy as np
import torch as th
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../include')
from utility import *
import exr
import hmc
import mfb

def main():

    ## generate synthetic image
    # d = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # n = 128
    # size = 21.21
    # camera = 25
    # f0 = 0.04

    # mfbObj = mfb.MFB(n, size, camera, f0, d)

    # para = [625, 0.42, 0.1, 0.1, 0.328, 3.8, 0.02, 7] 
    # # lgiht, albedo (r,g,b), rough, fsigma, fscale, iSigma
    
    # out = mfbObj.eval(para)
    # exr.write(out.detach().cpu().numpy(), 'out.exr')            


    ## run hmc
    d = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    n = 128
    size = 21.21
    camera = 25
    f0 = 0.04
    mfbObj = mfb.MFB(n, size, camera, f0, d)

    target = exr.read('out.exr')
    target = th.from_numpy(target).float().to(d)
    N = 1000
    hmcObj = hmc.HMCBumpTest(target, mfbObj, N)
    # hmcObj = hmc.HMCBump(target, mfbObj, N)
    para0 = np.array([0.2])
    hmcObj.sample(para0)
    xs = hmcObj.xs
    lpdfs = hmcObj.lpdfs

    minID = np.argmin(lpdfs)
    print(xs[minID,:])
    fig = plt.figure(figsize=(4,4))
    plt.hist(xs, bins=100)
    plt.show()
    


    # hmcObj.hist(n)
    # # ###
    # fig = plt.figure(figsize=(4,4))
    # plt.imshow(hmcObj.out) 
    # plt.axis('equal')
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':
    main()