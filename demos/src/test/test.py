import numpy as np
import torch as th
from forward import *
from gytools import *
from torch.autograd import Variable
from PIL import Image
import sumfunc
import matplotlib.pyplot as plt

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


####################################
fn_0 = 'test/target.png'
fn_1 = 'test/bad1.png'
fn_2 = 'test/good2.png'
im_0 = np.float32(np.array(Image.open(fn_0).resize((256,256))))/255
im_1 = np.float32(np.array(Image.open(fn_1).resize((256,256))))/255
im_2 = np.float32(np.array(Image.open(fn_2).resize((256,256))))/255

sumfuncObj = sumfunc.T_G(im_0, [0.001,0.001], device)

loss_1 = sumfuncObj.errloss(th.from_numpy(im_1).to(device))
loss_2 = sumfuncObj.errloss(th.from_numpy(im_2).to(device))

print(loss_1, loss_2)
