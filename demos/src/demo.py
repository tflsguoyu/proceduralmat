import os
import argparse
from datetime import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

from gytools import *
import exr
import hmc
import sumfunc
from forward_texture import *
from forward_bump import *
from forward_metal import *
from forward_flake import *
from forward_plaster_gy import *
from forward_leather_gy import *
from forward_wood_gy import *

#root_dir = '/home/guoyu/Documents/bayesian_appearance/hmc_pytorch/demos/'
root_dir = ''

def createForwardModel(args, isHDR, device):
    if args.forward == 'texture':
        return Texture(device)

    elif args.forward == 'bump':
        return Bump(args, 0.04, isHDR, device)

    elif args.forward == 'metal':
        return Metal(args, isHDR, device)

    elif args.forward == 'flake':
        return Flake(root_dir, args, isHDR, device)
        # return Flake('src/data', imres, size, camera, isHDR, device)

    elif args.forward == 'plaster':
        return Plaster(root_dir + 'data/', args, 0.04, isHDR, device)

    elif args.forward == 'leather':
        return Leather(root_dir + 'data/', args, 0.04, isHDR, device)

    elif args.forward == 'wood':
        return Wood(root_dir + 'data/', args, 0.04, isHDR, device)

def saveTexture(forwardObj, args):
    if args.forward == 'texture':
        assert(1==0)
    elif args.forward == 'bump':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal:', forwardObj.normal_tex.size())
        print('rough:', forwardObj.rough_tex.size())
        fn = os.path.join(args.in_dir, args.fn)[:-4]
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            fn + '_albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            fn + '_normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            fn + '_rough.exr')

    elif args.forward == 'metal':
        print('texture size:')
        print('normal:', forwardObj.normal_tex.size())
        print('roughx:', forwardObj.roughx_tex.size())
        print('roughy:', forwardObj.roughy_tex.size())
        print('f0:', forwardObj.f0_tex.size())
        fn = os.path.join(args.in_dir, args.fn)[:-4]
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            fn + '_normal.exr')
        exr.write(forwardObj.roughx_tex.detach().cpu().numpy(),
            fn + '_roughx.exr')
        exr.write(forwardObj.roughy_tex.detach().cpu().numpy(),
            fn + '_roughy.exr')
        exr.write(forwardObj.f0_tex.detach().cpu().numpy(),
            fn + '_f0.exr')

    elif args.forward == 'flake':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal_flake:', forwardObj.normal_flake_tex.size())
        print('rough_flake:', forwardObj.rough_flake_tex.size())
        print('f0_flake:', forwardObj.f0_flake_tex.size())
        print('rough_top:', forwardObj.rough_top_tex.size())
        fn = os.path.join(args.in_dir, args.fn)[:-4]
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            fn + '_albedo.exr')
        exr.write(forwardObj.normal_flake_tex.detach().cpu().numpy(),
            fn + '_normal_flake.exr')
        exr.write(forwardObj.rough_flake_tex.detach().cpu().numpy(),
            fn + '_rough_flake.exr')
        exr.write(forwardObj.f0_flake_tex.detach().cpu().numpy(),
            fn + '_f0_flake.exr')
        exr.write(forwardObj.rough_top_tex.detach().cpu().numpy(),
            fn + '_rough_top.exr')

    elif args.forward == 'plaster':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal:', forwardObj.normal_tex.size())
        print('rough:', forwardObj.rough_tex.size())
        fn = os.path.join(args.in_dir, args.fn)[:-4]
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            fn + '_albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            fn + '_normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            fn + '_rough.exr')

    elif args.forward == 'leather':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal:', forwardObj.normal_tex.size())
        print('rough:', forwardObj.rough_tex.size())
        fn = os.path.join(args.in_dir, args.fn)[:-4]
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            fn + '_albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            fn + '_normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            fn + '_rough.exr')

    elif args.forward == 'wood':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal:', forwardObj.normal_tex.size())
        print('rough:', forwardObj.rough_tex.size())
        fn = os.path.join(args.in_dir, args.fn)[:-4]
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            fn + '_albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            fn + '_normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            fn + '_rough.exr')

def main_generate(args):
    # generate synthetic image
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    fn_fullpath = os.path.join(args.in_dir, args.fn)[:-4]


    # initialize forward model
    forwardObj = createForwardModel(args, False, device)

    if args.para_all:
        para = np.array(args.para_all, dtype='float32')
    else:
        para = forwardObj.sample_prior()
    np.savetxt(fn_fullpath + '.csv', para, delimiter=" ", fmt='%.3f')
    np.set_printoptions(precision=3, suppress=True)
    print('para:', para.transpose())

    paraId = args.para_eval_idx
    forwardObj.loadPara(para, paraId)

    out = forwardObj.eval_render()
    outArray = out.detach().cpu().numpy()
    Image.fromarray(np.uint8(outArray*255)).save(fn_fullpath + '.png')

    if args.save_tex == 'yes':
        saveTexture(forwardObj, args)

def main_sample(args):
    # user defined parameters
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    fn_fullpath = os.path.join(args.in_dir, args.fn)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # load target image
    if fn_fullpath[-3:] == 'exr':
        isHDR = True
        target = exr.read(fn)
        assert(False) # not correct implemented
    else:
        isHDR = False
        target = Image.open(fn_fullpath)
        assert(target.width == target.height)
        target = target.resize((args.imres, args.imres)).convert('RGB')
        target = np.float32(np.array(target))/255

    # define summary function
    if args.sum_func == 'Bins':
        sumfuncObj = sumfunc.Bins(target, args.size, args.err_sigma, device)

    elif args.sum_func == 'Grids':
        sumfuncObj = sumfunc.Grids(target, args.err_sigma, device)

    elif args.sum_func == 'T_G':
        if isHDR:
            target = np.power(np.clip(target,0,1), 1/2.2)
            isHDR = False
        sumfuncObj = sumfunc.T_G(target, args.err_sigma, device)

    # initialize forward model
    forwardObj = createForwardModel(args, isHDR, device)

    para = np.array(args.para_all, dtype='float32')
    paraId = args.para_eval_idx
    forwardObj.loadPara(para, paraId)

    if args.sampleMethod == 'HMC':
        # initialize hmc
        hmcObj = hmc.HMC(forwardObj, sumfuncObj, args)
        xs, lpdfs = hmcObj.sample(args.epochs)
    elif args.sampleMethod == 'MALA':
        malaObj = hmc.MALA(forwardObj, sumfuncObj, args)
        xs, lpdfs = malaObj.sample(args.epochs)

    # xs = np.vstack(xs)
    # lpdfs = np.vstack(lpdfs)

    # np.savetxt(os.path.join(args.out_dir, 'sample.csv'),
    #     np.concatenate((lpdfs, xs), 1), delimiter=",")

def main_optimize(args):
    # user defined parameters
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    fn_fullpath = os.path.join(args.in_dir, args.fn)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # load target image
    assert(fn_fullpath[-3:] == 'png' or fn_fullpath[-3:] == 'jpg')
    isHDR = False
    target = Image.open(fn_fullpath)
    assert(target.width == target.height)
    target = target.resize((args.imres, args.imres), resample=Image.LANCZOS).convert('RGB')
    target = np.float32(np.array(target))/255

    # define summary function
    if args.sum_func == 'Bins':
        sumfuncObj = sumfunc.Bins(target, args.size, args.err_sigma, device)

    elif args.sum_func == 'Grids':
        sumfuncObj = sumfunc.Grids(target, args.err_sigma, device)

    elif args.sum_func == 'T_G':
        sumfuncObj = sumfunc.T_G(target, args.err_sigma, device)

    # initialize forward model
    forwardObj = createForwardModel(args, isHDR, device)

    para = np.array(args.para_all, dtype='float32')
    paraId = args.para_eval_idx
    forwardObj.loadPara(para, paraId)

    if args.forward == 'texture':
        print('############# is texture ###############')
        x0 = 0.5 + 0.2 * np.random.randn(args.imres, args.imres, 3).astype('float32')
        x0 = np.clip(x0, 0, 1)
        optimObj = hmc.Optim_Texture(forwardObj, sumfuncObj, args)
        losses = optimObj.optim(x0, args.epochs)
    else:
        # initialize optimizer
        optimObj = hmc.Optim(forwardObj, sumfuncObj, args, device)
        # optimization
        xs, losses = optimObj.optim(args.epochs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch HMC Sampling -- GY')
    parser.add_argument('--in_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--forward', required=True, help='bump|metal|...')
    parser.add_argument('--fn', required=True)
    parser.add_argument('--size', type=float, default=21.21)
    parser.add_argument('--camera', type=float, default=25)
    parser.add_argument('--para_all', type=float, nargs='+')
    parser.add_argument('--operation', default='sample', help='generate|optimize|sample(default)')
    parser.add_argument('--sampleMethod', default='HMC', help='HMC|MALA')
    parser.add_argument('--imres', type=int, default=256)
    parser.add_argument('--para_eval_idx', type=int, nargs='+')
    parser.add_argument('--sum_func', default='T_G', help='Bins|T_G(default)')
    parser.add_argument('--err_sigma', type=float, nargs='+')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lf_lens', type=float, default=0.04)
    parser.add_argument('--lf_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--diminish', type=float, default=0.5)
    parser.add_argument('--to_save', help='fig|(None)')
    parser.add_argument('--save_tex', default='no', help='yes|no')

    args = parser.parse_args()
    print(args)



    if args.operation == 'generate':
        main_generate(args)

    elif args.operation == 'sample':
        main_sample(args)

    elif args.operation == 'optimize':
        main_optimize(args)

    else:
        print('Current "--operation" does not support')
        exit()










# def errMap_bump(flag_sumFunc):
#     ## run hmc
#     device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

#     imsize = 21.21
#     camera = 25
#     f0 = 0.04

#     fn = '../in/target128.exr'
#     if fn[-3:] == 'exr':
#         isHDR = True
#         target = exr.read(fn)
#     else:
#         isHDR = False
#         target = np.float32(np.array(Image.open(fn)))/255

#     assert(target.shape[0] == target.shape[1])
#     imres = target.shape[0]

#     para   = np.array([    625, 0.42,  0.1,  0.1, 0.328, 1.4,   0.1,    7])
#     paraBd = np.array([[0,1000],[0,1],[0,1],[0,1],[0,1],[0,5],[0,0.2],[0,10]])
#     paraId = [5,6]

#     if flag_sumFunc == 'Bins':
#         sumfuncObj = sumfunc.Bins(target, imsize, device)

#     elif flag_sumFunc == 'T_G':
#         if isHDR:
#             target = np.power(np.clip(target,0,1), 1/2.2)
#             isHDR = False
#         sumfuncObj = sumfunc.T_G(target, device)

#     forwardObj = forward.Mfb(imres, imsize, camera, f0, para, paraBd, paraId, isHDR, device)

#     n = 100
#     out = np.zeros((n,n))
#     for i, fsigma in enumerate(np.linspace(0,5,n)):
#         print(i)
#         for j, fscale in enumerate(np.linspace(0,0.2,n)):
#             X = np.array([fsigma, fscale])
#             out[n-j-1, i] = sumfuncObj.logpdf(forwardObj.eval(X)).item()

#     print(np.min(out), np.max(out))
#     fig = plt.figure(figsize=(4,4))
#     plt.imshow(out, extent=[0,5,0,0.2], aspect="auto")
#     plt.colorbar()
#     plt.plot(1.4, 0.1, 'r*', alpha=1, markersize=5)
#     plt.savefig('../out/errMap_bump.png')
#     plt.show()

# def test(fn):
#     ## run hmc
#     device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

#     target = np.float32(np.array(Image.open(fn)))/255

#     assert(target.shape[0] == target.shape[1])
#     imres = target.shape[0]

#     sumfuncObj = sumfunc.T_G(target, device)
#     print(sumfuncObj.td_target)






    # generate('metal')
    # main('../in/bump_target_128.png', 5000, 'T_G', 'bump')
    # main('../in/metal_target_128.png', 5000, 'T_G', 'metal')
    # errMap_bump('T_G')
    # test('../in/gray.png')
    # test('../in/half.png')
    # main_texGen('../in/leather_bull.png', 'const', 10000, 0.01, 10)
    # main_texGen('../in/brick.png', '../in/brick_init.png', 10000, 2*np.pi/100, 10)
    # main_texGen('../in/rand1.png', '../in/rand0.png', 10000, 2*np.pi/100, 100)
