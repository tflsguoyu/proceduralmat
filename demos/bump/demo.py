from datetime import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../include')
from utility import *
import exr
import hmc
import forward
import sumfunc

def generateTargetImage():
    # generate synthetic image
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    imres = 128
    imsize = 21.21
    camera = 25
    f0 = 0.04

    forwardObj = forward.Mfb(imres, imsize, camera, f0, device)

    para = [625, 0.42, 0.1, 0.1, 0.328, 3.8, 0.02, 7] 
    # lgiht, albedo (r,g,b), rough, fsigma, fscale, iSigma
    para[0], para[1:4], para[4], para[5], para[6], para[7] = \
        forward.paraZip(para[0], para[1:4], para[4], para[5], para[6], para[7])

    out = forwardObj.eval(para)
    exr.write(out.detach().cpu().numpy(), 'target128.exr')            


def main():
    ## run hmc
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    imsize = 21.21
    camera = 25
    f0 = 0.04
    
    fn = 'target128.exr'
    N = 1000

    para = np.array([625, 0.42, 0.1, 0.1, 0.328, 3.0, 0.01, 7]) 
    # lgiht, albedo (r,g,b), rough, fsigma, fscale, iSigma
    para = forward.paraZip(para[0], para[1:4], para[4], para[5], para[6], para[7])
    paraIdx = [5,6]

    target = exr.read(fn)
    assert(target.shape[0] == target.shape[1])
    imres = target.shape[0]
    
    forwardObj = forward.Mfb(imres, imsize, camera, f0, device)
    sumfuncObj = sumfunc.Bins(target, imsize, device)

    # main
    hmcObj = hmc.HMC(forwardObj, sumfuncObj, N)

    now = datetime.now()
    print(now)
    hmcObj.sample(para, paraIdx)
    now = datetime.now()
    print(now)

    print('Reject: %d, outBound: Nan' % (hmcObj.num_reject))    

    # ###
    # fig = plt.figure(figsize=(4,4))
    # plt.hist(hmcObj.xs, bins=100)
    # plt.savefig('%s_rough_%d.png' % (fn[:-4], N))
    # print('DONE!!!')
    # plt.show()
 
    fig = plt.figure(figsize=(8,4))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.hist(hmcObj.xs[:,i], bins=100)
    plt.savefig('%s_rough_%d.png' % (fn[:-4], N))
    print('DONE!!!')
    plt.show()

    # fig = plt.figure(figsize=(12,4))
    # for i in range(3):
    #     plt.subplot(1,3,i+1)
    #     plt.hist(hmcObj.xs[:,i], bins=100)
    # plt.savefig('%s_rough_%d.png' % (fn[:-4], N))
    # print('DONE!!!')
    # plt.show()


if __name__ == '__main__':
    # generateTargetImage()
    main()