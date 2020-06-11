from util import *
sys.path.insert(1, 'src/forward/')
from bump import Bump
from metal import Metal
from flake import Flake
from plaster import Plaster
from leather import Leather
from wood import Wood
import torch.nn as nn
import torch.nn.functional as F
import shutil
import time

def createForwardModel(args, device):
    root_dir = ''
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

class HuNet(nn.Module):
    def __init__(self, N, device):
        super(HuNet, self).__init__()
        self.net = nn.Sequential(
            # input is 3 x 128 x 128
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # state size. 32 x 64 x 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # state size. 256 x 8 x 8
            nn.Flatten(),
            nn.Linear(256*7*7, 4096),
            nn.Dropout(inplace=True),
            # state size. 4096
            nn.Linear(4096, 256),
            nn.Dropout(inplace=True),
            # state size. 256
            nn.Linear(256, N),
            nn.Tanh()
            # state size. N
        ).to(device)

    def forward(self, x):
        y = self.net(x)
        return y

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    th.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[0:-8] + 'best.pth')

def logit(u):
    u = u.clamp(eps, 1-eps)
    return (u/(1-u)).log()

def sigmoid(v):
    return 1/(1+(-v).exp())

def plotAndSave(loss1, loss2, save_dir):
    plt.figure(figsize=(8,4))
    plt.plot(loss1, label='training loss')
    plt.plot(loss2, label='validation loss')
    plt.legend()
    plt.savefig(save_dir)
    plt.close()

def train(args, device):

    forwardObj = createForwardModel(args, device)
    para = np.array(args.para, dtype='float32')
    paraId = args.para_idx
    forwardObj.loadPara(para, paraId)
    # print('Here'); exit()

    net = HuNet(args.num_params, device)
    print(net)

    criterion = th.nn.MSELoss()
    print(criterion)

    optimizer = th.optim.Adam(net.parameters(), lr=args.lr)
    print(optimizer)

    best_loss = 999
    train_loss_list = []
    val_loss_list = []
    if args.resume:
        if os.path.isfile(args.resume):
            print("GY: loading checkpoint '{}'".format(args.resume))
            checkpoint = th.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_loss_list = checkpoint['train_loss']
            val_loss_list = checkpoint['val_loss']
            print("GY: loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("GY: can not load checkpoint from '{}'".format(args.resume))
            exit()

    N = args.iters_val
    img = th.zeros(N, 3, args.imres, args.imres, dtype=th.float32, device=device)
    for epoch in range(args.start_epoch, args.epochs):
        # train
        net = net.train()
        for i in range(int(args.iters_train/N)):
            # generate random parameters
            theta_inf = th.randn(N, args.num_params, device=device)
            theta_ref = sigmoid(theta_inf) * 2 -1
            # theta_inf = logit(th.rand(N, args.num_params, device=device))
            # theta_ref = sigmoid(theta_inf) * 2 -1

            # generate anchor image
            for j in range(N):
                img_this = forwardObj.eval_render(theta_inf[j])
                # gyArray2PIL(gyTensor2Array(img_this)).save(args.out_dir + 'tmp_%02d.png' % j)
                img[j] = img_this.permute(2,0,1)

            theta_pre = net(img)

            # eval triplet loss
            loss = criterion(theta_ref, theta_pre)

            optimizer.zero_grad()
            loss.backward()
            # for para in net.parameters():
            #     print(para.grad)
            # exit()
            optimizer.step()

        train_loss = loss.item()
        train_loss_list.append(train_loss)

        # validation
        net = net.eval()
        # generate random parameters
        theta_inf = th.randn(args.iters_val, args.num_params, device=device)
        theta_ref = sigmoid(theta_inf) * 2 -1
        # theta_inf = logit(th.randn(args.iters_val, args.num_params, device=device))
        # theta_ref = sigmoid(theta_inf) * 2 -1
        # generate anchor image
        for j in range(args.iters_val):
            img_this = forwardObj.eval_render(theta_inf[j])
            # gyArray2PIL(gyTensor2Array(img_this)).save(args.out_dir + 'tmp_%02d.png' % j)
            img[j] = img_this.permute(2,0,1)
        theta_pre = net(img)
        # eval triplet loss
        loss = criterion(theta_ref, theta_pre)
        val_loss = loss.item()
        val_loss_list.append(val_loss)

        # print and save
        print('GY: Epoch %03d: train loss, %.02f, val loss, %.02f' % (epoch+1, train_loss, val_loss))
        plotAndSave(np.vstack(train_loss_list), np.vstack(val_loss_list), args.out_dir+'loss.png')

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'train_loss': train_loss_list,
            'val_loss': val_loss_list,
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.out_dir + 'checkpoint_%04d.pth' % (epoch+1))

def loadImageToTensor(dir, res):
    print('Load target image from ', dir)
    target = Image.open(dir).convert('RGB')
    if not target.width == res:
        target = target.resize((res, res), Image.LANCZOS)
    target = gyPIL2Array(target)
    target = th.from_numpy(target).permute(2,0,1)
    return target

def test(args, device):

    img = loadImageToTensor(args.in_dir, args.imres).unsqueeze(0).to(device)

    args.imres = 256
    forwardObj = createForwardModel(args, device)
    para = np.array(args.para, dtype='float32')
    paraId = args.para_idx
    forwardObj.loadPara(para, paraId)
    # print('Here'); exit()

    net = HuNet(args.num_params, device)
    print(net)

    if args.resume:
        if os.path.isfile(args.resume):
            print("GY: loading checkpoint '{}'".format(args.resume))
            checkpoint = th.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            print("GY: loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("GY: can not load checkpoint from '{}'".format(args.resume))
            exit()
    else:
        print("GY: use --resume to load checkpoint")
        exit()


    net = net.eval()
    theta_pre = net(img)
    theta_pre = logit((theta_pre+1)/2)
    img_this = forwardObj.eval_render(theta_pre[0])
    d = datetime.now()
    gyArray2PIL(gyTensor2Array(img_this)).save(args.out_dir + 'tmp_%02d%02d%02d.png' % \
        (getattr(d, 'hour'),getattr(d, 'minute'),getattr(d, 'second')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=float, default=21.21)
    parser.add_argument('--camera', type=float, default=25)
    parser.add_argument('--imres', type=int, default=128)
    parser.add_argument('--operation', default='train', help='train|test')
    parser.add_argument('--forward', default='bump', help='bump|leather')
    parser.add_argument('--iters_train', type=int, default=100)
    parser.add_argument('--iters_val', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--in_dir', default='hu2019/in/')
    parser.add_argument('--out_dir', default='hu2019/out/')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--para', type=float, nargs='+')
    parser.add_argument('--para_idx', type=int, nargs='+')
    parser.add_argument('--save_tex', action='store_true')
    parser.add_argument('--resume', default=None)

    args = parser.parse_args()
    args.num_params = len(args.para_idx)
    print(args)

    gyCreateFolder(args.out_dir)

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    th.manual_seed(args.seed)

    if th.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = th.device("cuda:0" if args.cuda else "cpu")

    if args.operation == 'train':
        train(args, device)
    elif args.operation == 'test':
        test(args, device)

