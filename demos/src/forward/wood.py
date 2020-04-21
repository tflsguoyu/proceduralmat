from base import *
from lookup import *

def Lookup1d(A, x):
    return lookup1d(A, x).squeeze()

def Lookup2d(A, x, y):
    return lookup2d(A, th.stack((x, y))).squeeze()

def Lookup3d(A, p):
    return lookup2d(A, p.permute(2,0,1)).squeeze()

# all units in cm
class Wood(Material):
    def __init__(self, datadir, args, device):
        print('Initial Class forward::Wood()')
        super().__init__(args, device)
        self.save_tex = args.save_tex
        self.noise1d = th.rand(100, 1, device=device)
        self.noise2d = th.rand(100, 100, 1, device=device)
        self.noise3d = th.rand(20, 20, 3, device=device)

        self.paraPr = th.tensor([[1500, 500,   0,2500], # light
                                 [0.4, 0.3,0.01,0.95], # albedo_r
                                 [0.2, 0.2,0.01,0.95], # albedo_g
                                 [0.1, 0.1,0.01,0.95], # albedo_b
                                 [-5,  10,  -50,  50], # center_x
                                 [ -5,  10, -50,  50], # center_y
                                 [  0,  10,  -50,  50], # center_z
                                 [0.1, 0.2,   0,   1], # ring_size
                                 [2.5,   1, 1.2,   5], # lw_power
                                 [0.5, 0.2, 0.1, 0.9], # lw_fraction
                                 [0.2, 0.05, 0.01, 0.4],   # ew_ramp_width
                                 [0.2, 0.05, 0.01, 0.4],   # lw_ramp_width
                                 [10, 5, 1, 50],           # ssn_scale
                                 [0.5, 0.2, 0, 2],         # ssn_power
                                 [0.5, 0.3, 0, 2],         # grn_scale
                                 [0.5, 0.3, 0, 2],         # grn_amplitude
                                 [0.3, 0.2, 0, 1],         # gdn_scale
                                 [0.5, 0.3, 0, 2],         # gdn_amplitude
                                 [5, 10,  -90,   90],         # cut_angle
                                 [0.8, 0.1, 0.1, 0.8],     # rough
                                 [0.8, 0.05, 0.01, 0.8],       # ew_rough
                                 [0.005, 0.002, 0, 0.01],   # lw_height
                                 [ 15,   2,   0,  20]  # iSigma
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.paraNm = ['light','albedo','center','ring_size','lw_power','lw_fraction',
                       'ew_ramp_width','lw_ramp_width','ssn_scale','ssn_power',
                       'grn_scale','grn_amplitude','gdn_scale','gdn_amplitude',
                       'cut_angle','rough','ew_rough','lw_height','iSigma']

        self.noise = th.tensor(gyPIL2Array(Image.open(datadir + 'noise_1.png'))).to(device)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[0]),0)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[1]),1)

    def mk_plane(self, angle):
        x = angle * np.pi / 180
        c, s = x.cos(), x.sin()
        right = th.tensor([1, 0, 0], device=self.device)
        up = th.zeros(3, device=self.device)
        up[1] = s
        up[2] = c
        # up = [0, s, c]
        return right, up

    def eval_render(self, x=None):
        th.autograd.set_detect_anomaly(True)
        light, albedo, center, ring_size, lw_power, lw_fraction, ew_ramp_width, lw_ramp_width, ssn_scale, ssn_power, grn_scale, grn_amplitude, gdn_scale, gdn_amplitude, cut_angle, rough, ew_rough, lw_height, iSigma = self.unpack(x)

        s = self.size / 2
        px = th.linspace(-s, s, self.n, device=self.device)
        py, px = th.meshgrid(px, px)

        X, Y = self.mk_plane(cut_angle)
        p = gyDstack(px, X) + gyDstack(py, Y) + gyDstack(th.ones_like(px, device=self.device), center)

        # apply deformation to p
        p1 = p + gdn_amplitude * Lookup3d(self.noise3d, p * gdn_scale)

        x = p1[:, :, 0]
        y = p1[:, :, 1]
        r = th.sqrt(x**2 + y**2) # distance from tree center

        r1 = r + grn_amplitude * Lookup1d(self.noise1d, r)

        # compute latewood ratio
        t = r1/ring_size - (r1/ring_size).floor()
        # t = r1.fmod(0.5) / 0.5
        # t = r.fmod(ring_size) / ring_size
        ramp1 = t / ew_ramp_width
        ramp2 = (lw_fraction - t) / lw_ramp_width + 1
        lwr = th.min(ramp1, ramp2).clamp(0, 1)

        # compute final color
        lw_color = albedo ** lw_power
        color_tmp = gyDstack(1-lwr, albedo) + gyDstack(lwr, lw_color)

        noise_power = Lookup2d(self.noise2d, x * ssn_scale, y * ssn_scale) * ssn_power + 1
        color = color_tmp ** th.stack([noise_power] * 3, dim=2)

        # # render
        hf = lwr * lw_height
        hf += self.noise[:self.n, :self.n] * 0.005
        normal = gyHeight2Normal(hf, self.size / self.n)
        rough = ((1 - lwr) * ew_rough + rough).clamp(0,1)
        geom, n_dot_h = self.computeGeomTerm(normal)

        specular = geom * self.brdf(n_dot_h, rough**2, 0.04)
        specular = specular.unsqueeze(2).repeat(1,1,3)
        geom = geom.unsqueeze(2).repeat(1,1,3)
        diffuse = geom * color / np.pi
        img = (diffuse + specular) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0).clamp(min=eps))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        if self.save_tex == 'yes':
            self.albedo_tex = color
            self.normal_tex = (normal + 1) / 2
            self.rough_tex = rough ** 2

        return img
