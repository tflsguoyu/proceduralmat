from torchvision.utils import make_grid
from forward_plaster import *
    
def main_gen_grid():
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    forwardObj = Plaster(256, device)
    
    imgs = []

    for i in range(16):
        para = forwardObj.sample_prior()
        np.set_printoptions(precision=3, suppress=True)
        print('para:', para.transpose())   
        forwardObj.loadPara(para)   
        img = forwardObj.eval_render()
        imgs.append(img.permute(2,0,1))

    grid = make_grid(imgs, 4).permute(1,2,0)
    Image.fromarray(np.uint8(grid.detach().cpu().numpy()*255)).save('plaster.png')  

if __name__ == '__main__':
    main_gen_grid()
