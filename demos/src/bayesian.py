from util import *
sys.path.insert(1, 'src/forward/')
from bump import Bump
from metal import Metal
from flake import Flake
from plaster import Plaster
from leather import Leather
from wood import Wood
import mcmc
import sumfunc
import exr
np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.3f}'.format})

root_dir = ''

def createForwardModel(args, device):
    if args.forward == 'texture':
        return Texture(device)

    elif args.forward == 'bump':
        return Bump(args, device)

    elif args.forward == 'metal':
        return Metal(args, device)

    elif args.forward == 'flake':
        return Flake(root_dir + 'data/', args, device)

    elif args.forward == 'plaster':
        return Plaster(root_dir + 'data/', args, device)

    elif args.forward == 'leather':
        return Leather(root_dir + 'data/', args, device)

    elif args.forward == 'wood':
        return Wood(root_dir + 'data/', args, device)

def saveTexture(forwardObj, args):
    if args.forward == 'texture':
        assert(1==0)
    elif args.forward == 'bump':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal:', forwardObj.normal_tex.size())
        print('rough:', forwardObj.rough_tex.size())
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            args.out_dir + 'albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            args.out_dir + 'normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            args.out_dir + 'rough.exr')

    elif args.forward == 'metal':
        print('texture size:')
        print('normal:', forwardObj.normal_tex.size())
        print('roughx:', forwardObj.roughx_tex.size())
        print('roughy:', forwardObj.roughy_tex.size())
        print('f0:', forwardObj.f0_tex.size())
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            args.out_dir + 'normal.exr')
        exr.write(forwardObj.roughx_tex.detach().cpu().numpy(),
            args.out_dir + 'roughx.exr')
        exr.write(forwardObj.roughy_tex.detach().cpu().numpy(),
            args.out_dir + 'roughy.exr')
        exr.write(forwardObj.f0_tex.detach().cpu().numpy(),
            args.out_dir + 'f0.exr')

    elif args.forward == 'flake':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal_flake:', forwardObj.normal_flake_tex.size())
        print('rough_flake:', forwardObj.rough_flake_tex.size())
        print('f0_flake:', forwardObj.f0_flake_tex.size())
        print('rough_top:', forwardObj.rough_top_tex.size())
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            args.out_dir + 'albedo.exr')
        exr.write(forwardObj.normal_flake_tex.detach().cpu().numpy(),
            args.out_dir + 'normal_flake.exr')
        exr.write(forwardObj.rough_flake_tex.detach().cpu().numpy(),
            args.out_dir + 'rough_flake.exr')
        exr.write(forwardObj.f0_flake_tex.detach().cpu().numpy(),
            args.out_dir + 'f0_flake.exr')
        exr.write(forwardObj.rough_top_tex.detach().cpu().numpy(),
            args.out_dir + 'rough_top.exr')

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
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            args.out_dir + 'albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            args.out_dir + 'normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            args.out_dir + 'rough.exr')

    elif args.forward == 'wood':
        print('texture size:')
        print('albedo:', forwardObj.albedo_tex.size())
        print('normal:', forwardObj.normal_tex.size())
        print('rough:', forwardObj.rough_tex.size())
        exr.write(forwardObj.albedo_tex.detach().cpu().numpy(),
            args.out_dir + 'albedo.exr')
        exr.write(forwardObj.normal_tex.detach().cpu().numpy(),
            args.out_dir + 'normal.exr')
        exr.write(forwardObj.rough_tex.detach().cpu().numpy(),
            args.out_dir + 'rough.exr')

def main_generate(args):
    gyCreateFolder(args.out_dir)
    # generate synthetic image
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # initialize forward model
    forwardObj = createForwardModel(args, device)

    if args.para_all:
        para = np.array(args.para_all, dtype='float32')
    else:
        para = forwardObj.sample_prior()
    np.set_printoptions(precision=3, suppress=True)
    np.savetxt(args.out_dir + 'input.csv', para, delimiter=" ", fmt='%.3f')
    print('para:', para.transpose())

    paraId = args.para_eval_idx
    forwardObj.loadPara(para, paraId)

    out = forwardObj.eval_render()
    outArray = out.detach().cpu().numpy()
    Image.fromarray(np.uint8(outArray*255)).save(args.out_dir + 'input.png')

    if args.save_tex == 'yes':
        saveTexture(forwardObj, args)

def main_sample(args):
    gyCreateFolder(args.out_dir)
    save_args(args, args.out_dir + 'args.txt')

    # user defined parameters
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # load target image
    target = Image.open(args.in_dir + 'input.png')
    assert(target.width == target.height)
    target = target.resize((args.imres, args.imres), resample=Image.LANCZOS).convert('RGB')
    target = np.float32(np.array(target))/255

    # define summary function
    if args.sum_func == 'Bins':
        sumfuncObj = sumfunc.Bins(target, args.size, args.err_sigma, args.useFFT, device)

    elif args.sum_func == 'Grids':
        sumfuncObj = sumfunc.Grids(target, args.err_sigma, device)

    elif args.sum_func == 'T_G':
        sumfuncObj = sumfunc.T_G(target, args.err_sigma, device)

    elif args.sum_func == 'Mean_Var':
        sumfuncObj = sumfunc.Mean_Var(target, args.err_sigma, device)


    # initialize forward model
    forwardObj = createForwardModel(args, device)

    if args.para_all:
        para = np.array(args.para_all, dtype='float32')
        print('Start from manual initialization:', para)
    else:
        para = []
        for p in forwardObj.paraPr:
            para.append(p[0].item())
        para = np.hstack(para)
        print('Start from default initialization:', para)

    paraId = args.para_eval_idx
    forwardObj.loadPara(para, paraId)

    if args.mcmc == 'HMC':
        hmcObj = mcmc.HMC(forwardObj, sumfuncObj, args)
        xs, lpdfs = hmcObj.sample(args.epochs)
    elif args.mcmc == 'MALA':
        malaObj = mcmc.MALA(forwardObj, sumfuncObj, args)
        xs, lpdfs = malaObj.sample(args.epochs)

def main_optimize(args):
    gyCreateFolder(args.out_dir)
    save_args(args, args.out_dir + 'args.txt')

    # user defined parameters
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # load target image
    target = Image.open(args.in_dir + 'input.png')
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
    forwardObj = createForwardModel(args, device)

    if args.para_all:
        para = np.array(args.para_all, dtype='float32')
        print('Start from manual initialization:', para)
    else:
        para = []
        for p in forwardObj.paraPr:
            para.append(p[0].item())
        para = np.hstack(para)
        print('Start from default initialization:', para)

    paraId = args.para_eval_idx
    forwardObj.loadPara(para, paraId)

    if args.forward == 'texture':
        print('############# is texture ###############')
        x0 = 0.5 + 0.2 * np.random.randn(args.imres, args.imres, 3).astype('float32')
        x0 = np.clip(x0, 0, 1)
        optimObj = mcmc.Optim_Texture(forwardObj, sumfuncObj, args)
        losses = optimObj.optim(x0, args.epochs)
    else:
        # initialize optimizer
        optimObj = mcmc.Optim(forwardObj, sumfuncObj, args, device)
        # optimization
        xs, losses = optimObj.optim(args.epochs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian -- GY')
    parser.add_argument('--in_dir', default='')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--forward', required=True, help='bump|metal|...')
    parser.add_argument('--size', type=float, default=21.21)
    parser.add_argument('--camera', type=float, default=25)
    parser.add_argument('--para_all', type=float, nargs='+')
    parser.add_argument('--operation', default='sample', help='generate|optimize|sample(default)')
    parser.add_argument('--mcmc', default='HMC', help='HMC|MALA')
    parser.add_argument('--imres', type=int, default=256)
    parser.add_argument('--para_eval_idx', type=int, nargs='+')
    parser.add_argument('--sum_func', default='T_G', help='Bins|T_G(default)')
    parser.add_argument('--err_sigma', type=float, nargs='+')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lf_lens', type=float, default=0.04)
    parser.add_argument('--lf_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--to_save', help='fig|(None)')
    parser.add_argument('--save_tex', default='no', help='yes|no')
    parser.add_argument('--useFFT', action='store_true')

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
