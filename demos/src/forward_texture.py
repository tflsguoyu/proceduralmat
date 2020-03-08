import torch as th
import numpy as np
from gytools import *

class Texture:
    def __init__(self, device):
        print('Initial Class forward::Texture()')
        self.device = device

    def eval_render(self, x):
        return x.clamp(0.0, 1.0)

    def eval_img_laplacian(self, f):
        fc = f[1:-1, 1:-1]
        fl = 4 * fc - f[1:-1, 2:] - f[1:-1, :-2] - f[2:, 1:-1] - f[:-2, 1:-1]
        return (fl ** 2).mean()
        