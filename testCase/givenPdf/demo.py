from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../include')
from utility import KL_div 
import hmc
import forward

def genHist2D(xs, res):
    out = np.zeros((res,res), dtype='float32')
    N = len(xs)
    for k in range(N):
        j = int(np.round(xs[k,0] * res - 0.5))
        i = int(np.round(xs[k,1] * res - 0.5))
        out[i,j] += 1
    out /= N
    return out

def main():
    fn = 'vortex.exr'
    N = 100000
    x0 = np.array([0.5,0.5])
    
    forwardObj = forward.Forward(fn)
    forwardObj.comDiff()

    # main
    hmcObj = hmc.HMC(forwardObj, N)
    
    now = datetime.now()
    print(now)
    hmcObj.sample(x0)
    now = datetime.now()
    print(now)

    print('Reject: %d, outBound: %d' % (hmcObj.num_reject, hmcObj.num_outBound))
    
    pdfSampled = genHist2D(hmcObj.xs, forwardObj.imRes)
    kl_div = KL_div(forwardObj.pdf, pdfSampled)

    # ###
    fig = plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.imshow(forwardObj.pdf, vmin=0, vmax=max(forwardObj.pdf.flatten()))
    plt.axis('equal')
    plt.axis('off')
    plt.title('Target pdf (%dx%d)' % (forwardObj.imRes, forwardObj.imRes))

    plt.subplot(122)
    plt.imshow(pdfSampled, vmin=0, vmax=max(forwardObj.pdf.flatten())) 
    plt.axis('equal')
    plt.axis('off')
    plt.title('HMC (%dk|%.2fk,%.2fk|%.2f)' % \
        (hmcObj.N/1000, hmcObj.num_reject/1000, hmcObj.num_outBound/1000, kl_div))

    plt.savefig('%s_%d.png' % (fn[:-4], N))
    print('DONE!!!')
    plt.show()


if __name__ == '__main__':
    main()