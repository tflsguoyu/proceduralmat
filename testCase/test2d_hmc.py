import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../include')
import exr


fn = 'test2d/vortex/vortex'
dist = exr.read(fn + '.exr')

imgH,imgW = np.shape(dist)
assert(imgH==imgW)
imRes = imgH

imSize = 1
pixelUnit = imSize/imRes
lb = 0
ub = imSize

diff = np.zeros((imRes,imRes,2), dtype='float32')
for i in range(imRes):
    for j in range(imRes):
        if j > 0 and j < imRes-1:
            diff[i,j,0] = (dist[i,j+1] - dist[i,j-1]) / (2*pixelUnit)
        elif j == 0:
            diff[i,j,0] = (dist[i,j+1] - dist[i,j])   / (  pixelUnit)
        elif j == imRes-1:
            diff[i,j,0] = (dist[i,j]   - dist[i,j-1]) / (  pixelUnit)

        if i > 0 and i < imRes-1:
            diff[i,j,1] = (dist[i+1,j] - dist[i-1,j]) / (2*pixelUnit)
        elif i == 0:
            diff[i,j,1] = (dist[i+1,j] - dist[i,j])   / (  pixelUnit)
        elif i == imRes-1:
            diff[i,j,1] = (dist[i,j]   - dist[i-1,j]) / (  pixelUnit)

exr.write(diff[:,:,0], fn+'_diffx.exr')
exr.write(diff[:,:,1], fn+'_diffy.exr')

# step size
delta = 0.01
nSamples = 1e5
L = 5
epsilon = 1e-6

# func
def U(x):
    if x[0] <= 0 or x[1] <= 0 or x[0] >= 1 or x[1] >= 1:
        return 0
    j = int(np.round(x[0]*imRes-0.5))
    i = int(np.round(x[1]*imRes-0.5))
    return -np.log(max(epsilon, dist[i, j]))

def dU(x):
    if x[0] <= 0 or x[1] <= 0 or x[0] >= 1 or x[1] >= 1:
        return 0
    # print(x)
    j = int(np.round(x[0]*imRes-0.5))
    i = int(np.round(x[1]*imRes-0.5))
    return -diff[i, j, :] / max(epsilon, dist[i, j])


def K(p): return 0.5* np.dot(p,p)

def dK(p): return p

def leapfrog(x0, p0):
    p = p0 - delta/2 * dU(x0)
    x = x0 + delta   * dK(p)

    for i in range(L-1):
        p = p - delta * dU(x)
        x = x + delta * dK(p)

    p = p - delta/2 * dU(x)

    return x, p

# main
x0 = np.array((0.5, 0.5))
xs = [x0.copy()]
num_reject = 0
num_outBound = 0

while len(xs) < nSamples:
    if (len(xs)+1)%1000==0: print('%d/%d' % (len(xs)+1,nSamples))
    p0 = np.random.randn(2)
    x, p = leapfrog(x0, p0)
    if x[0] >= lb and x[1] >= lb and x[0] <= ub and x[1] <= ub:
        H0 = U(x0) + K(p0)
        H  = U(x)  + K(p)
        alpha = min(1, np.exp(H0 - H))
        if np.random.rand() < alpha:
            xs.append(x.copy())
            x0 = x
        else:
            num_reject += 1
    else:
        num_outBound += 1

print(num_reject)
xs = np.vstack(xs)

out = np.zeros((imRes,imRes), dtype='float32')
for k in range(nSamples):
    j = int(np.round(xs[k,0] * imRes - 0.5))
    i = int(np.round(xs[k,1] * imRes - 0.5))
    out[i,j] += 1
out /= nSamples

print(max(dist.flatten()))
print(max(out.flatten()))

def KL_div(a, b):
    assert(np.shape(a) == np.shape(b))
    h,w = np.shape(a)
    kl_div = 0
    for i in range(h):
        for j in range(w):
            kl_div += a[i,j] * np.log(max(epsilon,a[i,j]) / max(epsilon,b[i,j]))
    return kl_div

kl_div = KL_div(dist, out)

###
fig = plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(dist, vmin=0, vmax=max(dist.flatten()))
plt.axis('equal')
plt.axis('off')
plt.title('Target pdf (%dx%d)' % (imRes, imRes))

plt.subplot(122)
plt.imshow(out, vmin=0, vmax=max(dist.flatten()))
plt.axis('equal')
plt.axis('off')
plt.title('HMC (%dk|%.2fk,%.2fk|%.2f)' % (nSamples/1000, num_reject/1000, num_outBound/1000, kl_div))

plt.savefig(fn+'%d.png' % nSamples)
plt.show()
