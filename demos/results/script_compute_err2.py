from util import *
sys.path.insert(1, 'PerceptualSimilarity/')
import models
from matplotlib import rc

# rc('font',**{'family':'Times New Roman'})
# activate latex text rendering
# from matplotlib import rc,rcParams
# rc('text', usetex=True)
# rc('font', size=20)
rc('font', family='serif')
rc('pdf', fonttype=42)

def loadImage(fn):
    im = Image.open(fn).convert('RGB')
    im = im.resize((256, 256), Image.LANCZOS)
    im = gyPIL2Array(im)
    im = th.from_numpy(im).permute(2,0,1)
    im = im*2-1
    im = im.unsqueeze(0)
    return im

if __name__ == '__main__':
    # os.remove('error_lpips.txt')
    # os.remove('error_mse.txt')

    criterion = th.nn.MSELoss().cuda()
    LPIPS = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    refs = th.zeros(1, 3, 256, 256)
    pres = th.zeros(1, 3, 256, 256)
    mat_list = gyListNames('results/*')
    N = len(mat_list)
    loss_lpips = np.zeros((3, N))
    xticks_name = []
    for i, mat in enumerate(mat_list):
        fn = 'results/' + mat + '/target.jpg'
        refs[0,:] = loadImage(fn)
        refs = refs.cuda()

        fn = 'results/' + mat + '/ours.jpg'
        pres[0,:] = loadImage(fn)
        pres_ours = pres.cuda()

        fn = 'results/' + mat + '/Hu.jpg'
        pres[0,:] = loadImage(fn)
        pres_hu = pres.cuda()

        fn = 'results/' + mat + '/egsr.jpg'
        pres[0,:] = loadImage(fn)
        pres_egsr = pres.cuda()

        loss_lpips[0,i]= (LPIPS.forward(refs, pres_ours).sum()).item()
        loss_lpips[1,i]= (LPIPS.forward(refs, pres_hu).sum()).item()
        loss_lpips[2,i]= (LPIPS.forward(refs, pres_egsr).sum()).item()

        xticks_name.append(mat[2:])

    print(xticks_name)
    print(loss_lpips)

    xticks_name = ['Bump-3', 'Bump-4', \
                   'Leather-3', 'Leather-4', 'Leather-5', 'Leather-6', \
                   'Plaster-3', 'Plaster-4', \
                   'Flake-3', 'Flake-4', \
                   'Metal-3', \
                   'Wood-3', 'Wood-4', 'Wood-5']

    plt.figure(figsize=(8,3))
    X = np.arange(0,N,1)
    print(X)
    plt.bar(X + 0.3, loss_lpips[0,:], color = [218/255,125/255,129/255], width = 0.2)
    plt.bar(X + 0.5, loss_lpips[1,:], color = [186/255,221/255,140/255], width = 0.2)
    plt.bar(X + 0.7, loss_lpips[2,:], color = [113/255,209/255,209/255], width = 0.2)
    plt.ylabel('LPIPS', fontsize=15)
    plt.xticks(X, xticks_name, fontsize=15, rotation=60)
    plt.ylim(0,0.8)
    plt.yticks(np.arange(0, 1.0, 0.5))
    plt.legend(labels=['Ours', '[HDR19]', '[DAD*18]'], loc=1, fontsize=13)
    plt.savefig('LPIPS.pdf',bbox_inches='tight')
    plt.close()

