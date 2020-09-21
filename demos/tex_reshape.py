import sys
sys.path.insert(1, 'src/')
from util import *

if __name__ == '__main__':

    in_dir = 'egsr2019/out/'
    # out_dir = 'egsr2019/out/'
    out_dir = '../paper/egsr19/'

    mat_list = gyListNames(in_dir+'*')
    for j, mat in enumerate(mat_list):
        if '1_bump_real1' == '1_bump_real1':
            print(mat)
            tex = Image.open(in_dir + mat + '/tex.png')
            tex1 = tex.crop((0,0,256*2,256))
            tex2 = tex.crop((256*2,0,256*4,256))
            gyConcatPIL_v(tex1, tex2).save(out_dir + mat + '/tex2x2.jpg')
            # break
