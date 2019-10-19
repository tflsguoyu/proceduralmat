from datetime import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image 
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

    para = np.array([625, 0.42, 0.1, 0.1, 0.328, 2.4, 0.1, 7])
    # light, albedo (r,g,b), rough, fsigma, fscale, iSigma
    para = forward.paraZip(para[0], para[1:4], para[4], para[5], para[6], para[7])

    out = forwardObj.eval(para)
    out = exr.read('target128_0.exr')
    sumfuncObj = sumfunc.TextureDescriptor(out, device)
    print(sumfuncObj.logpdf(forwardObj.eval(para)).item())

    # exr.write(out.detach().cpu().numpy(), 'target128_1.exr')
    # Image.fromarray(np.uint8(np.power(out.detach().cpu().numpy(), 1/2.2)*255)).save('target128_1.png')            


def main():
    ## run hmc
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    imsize = 21.21
    camera = 25
    f0 = 0.04
    
    fn = 'target128_1.exr'
    N = 5000

    para = np.array([625, 0.42, 0.1, 0.1, 0.328, 1, 0.05, 7]) 
    # lgiht, albedo (r,g,b), rough, fsigma, fscale, iSigma
    para = forward.paraZip(para[0], para[1:4], para[4], para[5], para[6], para[7])
    paraIdx = [5,6]

    target = exr.read(fn)
    assert(target.shape[0] == target.shape[1])
    imres = target.shape[0]
    
    forwardObj = forward.Mfb(imres, imsize, camera, f0, device)
    sumfuncObj = sumfunc.Bins(target, imsize, device)
    # sumfuncObj = sumfunc.TextureDescriptor(target, device)

    # main
    hmcObj = hmc.HMC(forwardObj, sumfuncObj, N)

    now = datetime.now()
    print(now)
    hmcObj.sample(para, paraIdx)
    now = datetime.now()
    print(now)

    print('Reject: %d, outBound: Nan' % (hmcObj.num_reject))    


    np.savetxt("xs.csv", hmcObj.xs, delimiter=",")

    # ###
    # fig = plt.figure(figsize=(4,4))
    # plt.hist(hmcObj.xs, bins=100)
    # plt.savefig('%s_rough_%d.png' % (fn[:-4], N))
    # print('DONE!!!')
    # plt.show()
 
    fig = plt.figure(figsize=(4,4))
    plt.hist2d(hmcObj.xs[:,0]*5, hmcObj.xs[:,1]*0.2, bins=100, norm=LogNorm())
    plt.plot(1.4, 0.1, 'r*', alpha=1, markersize=5);
        # for i in range(2):
    #     plt.subplot(1,2,i+1)
    #     plt.hist(hmcObj.xs[:,i], bins=100)
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