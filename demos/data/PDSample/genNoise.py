from os import system
import matplotlib.pyplot as plt
from sys import argv
from numpy import *
import numpy as np
from PIL import Image

cmd = "g++ PDSample.cpp PDSampling.cpp RangeList.cpp RNG.cpp ScallopedSector.cpp -o PDSample"
system(cmd)

res = 1024
n = res + 10*2
_r = 0.015
seed = 12345

cmd = "./PDSample -r %f -s %d" % (_r, seed)
system(cmd)

with open('pdsample.out', 'rb') as f:
	k = fromstring(f.read(4), int32, 1)[0]
	print(k)
	radius = fromstring(f.read(4), float32, 1)[0]
	print(radius)
	P = fromstring(f.read(k*8), float32, k*2)


P.shape = k, 2
# plt.plot(P[:,0], P[:,1], '.')
# plt.show()

# map to pixel units
r = _r
real2pix = n / 2.0
P += 1
P *= real2pix
r *= real2pix
r = int32(ceil(r * 2.5))
print('r:', r)

x = arange(2*r + 1, dtype=float32) - r
x, y = meshgrid(x, x)
dist = sqrt(x**2 + y**2)

# padded
d = ones((n + 2*r, n + 2*r), dtype=float32) * n
idx = zeros((n + 2*r, n + 2*r), dtype=int32)
P += r

def splat(ij, flake_id):
	i0, j0 = ij - r
	i1, j1 = ij + r + 1
	crop = d[i0 : i1, j0 : j1]
	mask = dist < crop
	crop[mask] = dist[mask]
	d[i0 : i1, j0 : j1] = crop
	crop = idx[i0 : i1, j0 : j1]
	crop[mask] = int32(flake_id)
	idx[i0 : i1, j0 : j1] = crop

for i in range(k):
	pt = int32(P[i, :])
	splat(pt, i+1)

d = d[r:-r, r:-r]
idx = idx[r:-r, r:-r]
d = d[10:10+res, 10:10+res]
# print('d shape:', np.array(d).max())
d = np.array(d)
d = np.clip(d, a_min=6,a_max=20)
d = (d-d.min()+2)/(d.max()-d.min()+2)
Image.fromarray(np.uint8((1-d)*255)).save('out.png')


# print(idx.min())
# plt.figure(figsize=(18,9))
# plt.subplot(121)
# plt.imshow(d)
# plt.colorbar()
# plt.title('distance to flake center')
# plt.subplot(122)
# plt.imshow(idx)
# plt.colorbar()
# plt.title('flake index')
# plt.savefig('pdsample_%.3f_%d.png' % (_r,seed), bbox_inches='tight')
# plt.show()
