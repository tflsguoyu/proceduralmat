from base import *
from lookup import *

class Leather(Material):
    def __init__(self, datadir, args, device):
        print('Initial Class forward::Leather()')
        super().__init__(args, device)
        self.op = args.operation
        self.save_tex = args.save_tex
        self.f0 = 0.04
        self.paraPr = th.tensor([[1500, 100,   0,2500], # light
                                 [0.4, 0.3,   0,   1], # albedo_r
                                 [0.4, 0.3,   0,   1], # albedo_g
                                 [0.4, 0.3,   0,   1], # albedo_b
                                 # [0.1, 0.1,   0,   1], # albedo_r
                                 # [0.1, 0.1,   0,   1], # albedo_g
                                 # [0.1, 0.1,   0,   1], # albedo_b
                                 [0.5, 0.3, 0.2, 0.8], # rough
                                 [0.2, 0.3,   0, 0.4], # rough_var
                                 [  1, 0.3,0.001,  2], # height
                                 [0.5, 0.3,   0,   2], # power
                                 [0.5, 0.3, 0.1,   1], # scale
                                 [0.01, 0.1, 0.001, 0.025], # noise scale
                                 [  0, 0.3,  -1,   1], # shiftx
                                 [  0, 0.3,  -1,   1], # shifty
                                 [ 15,   3,   5,  20]  # iSigma
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,1,1,1,1,1,1,2,1]
        self.paraNm = ['light','albedo','rough','rough_var','height','power','scale','noise_scale','shift','iSigma']

        self.noise = th.tensor(gyPIL2Array(Image.open(datadir + 'noise_1.png'))).to(device)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[0]),0)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[1]),1)

        self.cells = th.tensor(gyPIL2Array(Image.open(datadir + 'cell_1.png'))).to(device)
        # self.cells = self.cells.roll(np.random.randint(self.cells.size()[0]),0)
        # self.cells = self.cells.roll(np.random.randint(self.cells.size()[1]),1)

    def resizedCenterCrop(self, I0, s, t, n):
        n0 = I0.size()[0]
        if self.op == 'generate' or self.op == 'train':
            I0 = I0.roll(np.random.randint(n0),0)
            I0 = I0.roll(np.random.randint(n0),1)
            if np.random.rand() > 0.5:
                I0 = I0.transpose(0,1)
            if np.random.rand() > 0.5:
                I0 = I0.flip(0)
            if np.random.rand() > 0.5:
                I0 = I0.flip(1)

        v = th.arange(n, dtype=th.float32, device=self.device)
        i, j = th.meshgrid((v, v))
        P = th.stack([i, j], 0)

        t = t.unsqueeze(-1).unsqueeze(-1).expand_as(P)
        P0 = x2p((p2x(P,n)+t)*s, n0)
        # print(P0)
        return lookup2d(I0.unsqueeze(-1), P0).squeeze(-1)

    def eval_param_maps(self, rough, rough_var, height, power, scale, shift, noise_scale):
        sizePerPixel = float(self.size) / self.n

        base_cells = self.resizedCenterCrop(self.cells, scale, shift, self.n).clamp(eps,1-eps)
        noise = self.resizedCenterCrop(self.noise, scale, shift, self.n)

        # compute heightfield and turn to normals
        hf = (base_cells**power * height).clamp(eps,1.0) + (noise*2 - 1) * noise_scale
        normal = gyHeight2Normal(hf, sizePerPixel)

        # roughness depends on base_noise: increase in lower areas
        rough = (1-base_cells) * rough_var + rough

        return normal, rough.clamp(0.1, 0.8)

    def eval_render(self, x=None):
        light, albedo, rough, rough_var, height, power, scale, noise_scale, shift, iSigma = self.unpack(x)

        normal_cell, rough = self.eval_param_maps(rough, rough_var, height, power, scale, shift, noise_scale)
        geom_cell, n_dot_h_cell = self.computeGeomTerm(normal_cell)

        diffuse = gyDstack(geom_cell, albedo/np.pi)
        specular = geom_cell * self.brdf(n_dot_h_cell, rough.pow(2.0), self.f0)
        specular = specular.unsqueeze(2).repeat(1,1,3)
        img = (diffuse + specular) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0).clamp(min=eps))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        if self.save_tex == 'yes':
            self.albedo_tex = gyDstack(th.ones_like(geom_cell), albedo)
            self.normal_tex = (normal_cell + 1) / 2
            self.rough_tex = rough**2

        return img


        # print('light', light)
        # print('albedo', albedo)
        # print('rough', rough)
        # print('rough_var', rough_var)
        # print('height', height)
        # print('power', power)
        # print('scale', scale)
        # print('shift', shift)
        # print('iSigma', iSigma)
        # exit()
