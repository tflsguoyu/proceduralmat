import os
import sys
import glob
import json
import math
import time
import random
import argparse
import numpy as np
import torch as th
import torchvision
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable

eps = 1e-8

def save_args(args, dir):
    with open(dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def gyCreateFolder(dir):
    if not os.path.exists(dir):
        print("\ncreate directory: ", dir)
        os.makedirs(dir)

def gyListNames(in_dir):
    dir_list = sorted(glob.glob(in_dir))
    fn_list = []
    for dir in dir_list:
        fn_list.append(os.path.split(dir)[1])
    return fn_list

def gyConcatPIL_h(im1, im2):
    if im1 is None:
        return im2
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def gyConcatPIL_v(im1, im2):
    if im1 is None:
        return im2
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def gyPIL2Array(im):
    return np.array(im).astype(np.float32)/255

def gyPIL2Array16(im):
    return np.array(im).astype(np.float32)/65536

def gyArray2PIL(im):
    return Image.fromarray((im*255).astype(np.uint8))

def gyApplyGamma(im, gamma=1/2.2):
    if gamma < 1: im = im.clip(min=eps)
    return im**gamma

def gyApplyGammaPIL(im, gamma=1/2.2):
    return gyArray2PIL(gyApplyGamma(gyPIL2Array(im),gamma))

def gyTensor2Array(im):
    return im.detach().cpu().numpy()

def gyCreateThumbnail(fnA,w=128,h=128):
    fnB = os.path.join(os.path.split(fnA)[0], 'jpg')
    gyCreateFolder(fnB)
    fnB = os.path.join(fnB, os.path.split(fnA)[1][:-3]+'jpg')
    os.system('convert ' + fnA + ' -resize %dx%d -quality 100 ' % (w,h) + fnB)

def roll(dim, A, n):
    assert( A.dim() > dim );
    return th.cat((A.narrow(dim, n, A.size(dim) - n), A.narrow(dim, 0, n)), dim)

def roll0(A, n):
    return roll(0, A, n)

def roll1(A, n):
    return roll(1, A, n)

def gyShift(A):
    m = A.size(0)
    n = A.size(1)
    assert( m == n )
    nh = n // 2
    return roll1(roll0(A, nh), nh)

def gyNormalize(x):
    s = th.sqrt((x ** 2).sum(2).clamp(min=eps))
    return x / th.stack((s, s, s), 2)

def gyHeight2Normal(hf, pix_size):
    c = 1 / pix_size
    dx = roll1(hf, 1) - hf
    dy = hf - roll0(hf, 1)
    N = th.stack((-c*dx, -c*dy, th.ones_like(dx)), 2)
    N = gyNormalize(N)
    return N

def gyDstack(img, a):
    return th.stack([img * x for x in a], 2)

def gySampleVar(mu, sigma, mn, mx, num=1):
    x = th.randn(num) * sigma + mu
    return x.clamp(mn, mx) # FIXME
