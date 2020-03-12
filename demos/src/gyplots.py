import torch as th
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from imageio import mimsave
import glob

def plotOptimFrames(dir, losses, img, img0, i, N):
    def forceAspect(ax, aspect=1):
        extent = ax.get_window_extent()
        ax.set_aspect(abs((extent.y1-extent.y0)/(extent.x1/extent.x0)) * aspect)
    fig = plt.figure(figsize=(3*3, 3))
    plt.subplot(1,3,1)
    plt.imshow(img0)
    plt.axis('off')
    plt.title('Target')
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(img)
    plt.title('Iter #%d' % i)
    ax = fig.add_subplot(1,3,3)
    ax.plot(np.log(losses))
    forceAspect(ax,1)
    plt.xlim(0,N); plt.ylim(7.8,np.log(losses[0]))
    plt.title('Optim. loss (log)')
    plt.savefig(dir+'frame_%04d.jpg' % i)
    plt.close()
    # exit()

def plotFigureOptim(dir, losses, img, img0, i):
    if not os.path.exists(dir+'target.png'):
        Image.fromarray(np.uint8(img0*255)).save(dir+'target.png')
    Image.fromarray(np.uint8(img*255)).save(dir+'optim%04d.png' % i)

    fig = plt.figure(figsize=(2.56,2.56))
    plt.plot(losses)
    # plt.xlim(0,400); plt.ylim(0,0.013);
    plt.title('L2 loss of $T_G$')
    plt.savefig(dir+'loss%04d.png' % i)
    plt.close()

def png2gif_optim(dir):
    ims = []
    im_target = Image.open(dir+'target.png')
    w, h = im_target.size
    fn_optim_list = sorted(glob.glob(dir+'optim*.png'))
    fn_loss_list = sorted(glob.glob(dir+'loss*.png'))
    for i,fn in enumerate(fn_optim_list):
        im = get_concat_h(im_target, Image.open(fn))
        im = get_concat_h_resize(im, Image.open(fn_loss_list[i]))
        ims.append(np.array(im))
    mimsave(dir+'tmp.gif', ims, fps=10, loop=999)

def plotFigure(dir, losses, xs, img, img0, id, num_reject, time):
    if not os.path.exists(dir+'target.png'):
        Image.fromarray(np.uint8(img0*255)).save(dir+'target.png')
    Image.fromarray(np.uint8(img*255)).save(dir+'hmc%05d.png' % id)

    fig = plt.figure(figsize=(4,4))
    plt.plot(losses)
    plt.title('Logpdf of posterior')
    plt.savefig(dir+'loss%05d.png' % id)
    plt.close()

    xs = np.vstack(xs)
    C = xs.shape[1]
    # print(C-1)
    if C == 1:
        fig = plt.figure(figsize=(4,4))
        plt.hist(xs, bins=100)
    else:
        fig = plt.figure(figsize=(3*(C-1),3*(C-1)))
        for j in range(1,C):
            for i in range(j):
                k = i*(C-1) + j
                # print('i,j,k',i,j,k)
                plt.subplot(C-1,C-1,k)
                plt.hist2d(xs[:,i], xs[:,j], bins=50, norm=LogNorm())
                plt.plot(xs[-1,i], xs[-1,j], 'r.', markersize=20)
                # plt.xlim(0.095,0.115); plt.ylim(0.3,1.0)
                plt.xlabel('para%d' % (i+1), fontSize=10)
                plt.ylabel('para%d' % (j+1), fontSize=10)
                # plt.legend(prop={'size': 16})


        # M = (N+1)//2
        # fig = plt.figure(figsize=(4*M,8))
        # for i in range(N):
        #     plt.subplot(2,M,i+1)
        #     plt.hist2d(xs[:,i], xs[:,i+1], bins=100, norm=LogNorm())
        #     plt.plot(xs[-1,i], xs[-1,i+1], 'r.', markersize=20)
    plt.title('Accept %d, reject %d, time %ds' % (len(xs), num_reject, time))
    plt.tight_layout()
    plt.savefig(dir+'pdf%05d.png' % id)
    plt.close()

def png2gif(dir):
    ims = []
    im_target = Image.open(dir+'target.png')
    w, h = im_target.size
    fn_optim_list = sorted(glob.glob(dir+'hmc*.png'))
    fn_pdf_list = sorted(glob.glob(dir+'pdf*.png'))
    for i,fn in enumerate(fn_optim_list):
        im = get_concat_v(im_target, Image.open(fn))
        im = get_concat_h_resize(im, Image.open(fn_pdf_list[i]))
        ims.append(np.array(im))
    mimsave(dir+'tmp.gif', ims, fps=10, loop=999)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_concat_h_resize(im1, im2, resample=Image.LANCZOS, resize_big_image=True):
    if im1.height == im2.height:
        _im1 = im1
        _im2 = im2
    elif (((im1.height > im2.height) and resize_big_image) or
          ((im1.height < im2.height) and not resize_big_image)):
        _im1 = im1.resize((int(im1.width * im2.height / im1.height), im2.height), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((int(im2.width * im1.height / im2.height), im1.height), resample=resample)
    dst = Image.new('RGB', (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst

def get_concat_v_resize(im1, im2, resample=Image.LANCZOS, resize_big_image=True):
    if im1.width == im2.width:
        _im1 = im1
        _im2 = im2
    elif (((im1.width > im2.width) and resize_big_image) or
          ((im1.width < im2.width) and not resize_big_image)):
        _im1 = im1.resize((im2.width, int(im1.height * im2.width / im1.width)), resample=resample)
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize((im1.width, int(im2.height * im1.width / im2.width)), resample=resample)
    dst = Image.new('RGB', (_im1.width, _im1.height + _im2.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (0, _im1.height))
    return dst