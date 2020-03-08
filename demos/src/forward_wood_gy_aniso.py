from PIL import Image
from gytools import *
from forward import *
import lookup


def lookup1d(A, x):
    return lookup.lookup1d(A, x).squeeze()

def lookup2d(A, x, y):
    return lookup.lookup2d(A, th.stack((x, y))).squeeze()

def lookup3d(A, p):
    return lookup.lookup2d(A, p.permute(2,0,1)).squeeze()

# all units in cm
class Wood(Material):
    def __init__(self, datadir, args, f0, isHDR, device):
        print('Initial Class forward::Wood()')
        super().__init__(args.imres, args.size, args.camera, isHDR, device)
        self.save_tex = args.save_tex
        self.noise1d = th.rand(100, 1, device=device)
        self.noise2d = th.rand(100, 100, 1, device=device)
        self.noise3d = th.rand(20, 20, 3, device=device)

        self.paraPr = th.tensor([[999, 300,   0,2500], # light
                                 [0.6, 0.1,0.01,0.95], # albedo_r
                                 [0.4, 0.1,0.01,0.95], # albedo_g
                                 [0.1, 0.1,0.01,0.95], # albedo_b
                                 [  0,  5,  -50,  50], # center_x
                                 [  0,  10, -50,  50], # center_y
                                 [  0, 10,  -50,  50], # center_z
                                 [0.5, 0.2, 0.1,   1], # ring_size
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
                                 [5, 2,  0,   90],         # cut_angle
                                 [0.1, 0.05, 0.01, 0.3],       # ew_rough
                                 [0.005, 0.002, 0, 0.01],   # lw_height      
                                 [ 10,   2,   5,  15],  # iSigma
                                 [0.3, 0.2, 0.1,   0.8], # rough_x
                                 [0.3, 0.2, 0.1,   0.8], # rough_y
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.paraNm = ['light','albedo','center','ring_size','lw_power','lw_fraction',
                       'ew_ramp_width','lw_ramp_width','ssn_scale','ssn_power',
                       'grn_scale','grn_amplitude','gdn_scale','gdn_amplitude',
                       'cut_angle','ew_rough','lw_height','iSigma','roughx','roughy']

        self.noise = from16bit(Image.open(datadir + 'bnw-spots.png')).to(device)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[0]),0)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[1]),1)

        self.initPhase()

    def mk_plane(self, angle):
        x = angle * np.pi / 180
        c, s = x.cos(), x.sin()
        right = th.tensor([1, 0, 0], device=self.device)
        up = th.zeros(3, device=self.device)
        up[1] = s
        up[2] = c
        # up = [0, s, c]
        return right, up

    def initPhase(self):
        self.sizePerPixel = float(self.size) / self.n
        self.phase = 2 * np.pi * th.rand(self.n, self.n, device=self.device)

        vF = th.arange(self.n, dtype=th.float32, device=self.device)
        vF = ((vF + 0.5) / self.n - 0.5) / self.sizePerPixel
        self.yF, self.xF = th.meshgrid((vF, vF))


    def computeBumpHeight(self, fsigmax, fsigmay, fscale):
        amp = (-0.5 * ((self.xF/fsigmax.clamp(min=eps)).pow(2.0) + (-self.yF/fsigmay.clamp(min=eps)).pow(2.0))).exp()
        amp = shift(amp*fscale)
        profile = th.stack((amp*th.cos(self.phase), amp*th.sin(self.phase)), 2)
        hf = th.ifft(profile, 2)[:, :, 0] / (self.sizePerPixel**2)
        return hf

    def ggx_ndf_aniso(self, cos_h, cos_hx, cos_hy, alpha_x, alpha_y):
        denom = np.pi * alpha_x * alpha_y * \
            (cos_hx**2/(alpha_x**2).clamp(0,1) + cos_hy**2/(alpha_y**2).clamp(0,1) + cos_h**2)**2
        return 1.0 / denom.clamp(min=eps)

    def brdf_aniso(self, n_dot_h, x_dot_h, y_dot_h, alpha_x, alpha_y, f0):
        D = self.ggx_ndf_aniso(n_dot_h, x_dot_h, y_dot_h, alpha_x, alpha_y)
        return  D *f0 / (4 * (n_dot_h**2).clamp(eps,1))


    def computeTangent(self, normal):
        n1n3 = normal[:,:,0] / normal[:,:,2].clamp(0,1)
        tangent = th.stack((1.0/(n1n3.pow(2.0)+1.0).sqrt(),
                                th.zeros_like(n1n3),
                                -n1n3/(n1n3.pow(2.0)+1.0).sqrt()), 2)
        bitangent = th.cross(normal, tangent)
        x_dot_h = (self.omega*tangent).sum(2)
        y_dot_h = (self.omega*bitangent).sum(2)

        return x_dot_h, y_dot_h


    def eval_render(self, x=None):
        th.autograd.set_detect_anomaly(True)
        light, albedo, center, ring_size, lw_power, lw_fraction, ew_ramp_width, lw_ramp_width, ssn_scale, ssn_power, grn_scale, grn_amplitude, gdn_scale, gdn_amplitude, cut_angle, ew_rough, lw_height, iSigma, roughx, roughy= self.unpack(x)
        
        s = self.size / 2
        px = th.linspace(-s, s, self.n, device=self.device)
        py, px = th.meshgrid(px, px)

        X, Y = self.mk_plane(cut_angle)
        p = dstack(px, X) + dstack(py, Y) + dstack(th.ones_like(px, device=self.device), center)

        # apply deformation to p
        p1 = p + gdn_amplitude * lookup3d(self.noise3d, p * gdn_scale)

        x = p1[:, :, 0]
        y = p1[:, :, 1]
        r = th.sqrt(x**2 + y**2) # distance from tree center
        
        r1 = r + grn_amplitude * lookup1d(self.noise1d, r)
        
        # compute latewood ratio
        t = r1/ring_size - (r1/ring_size).floor() 
        # t = r1.fmod(0.5) / 0.5
        # t = r.fmod(ring_size) / ring_size
        ramp1 = t / ew_ramp_width
        ramp2 = (lw_fraction - t) / lw_ramp_width + 1
        lwr = th.min(ramp1, ramp2).clamp(0, 1)

        # compute final color
        lw_color = albedo ** lw_power
        color_tmp = dstack(1-lwr, albedo) + dstack(lwr, lw_color)

        noise_power = lookup2d(self.noise2d, x * ssn_scale, y * ssn_scale) * ssn_power + 1
        color = color_tmp ** th.stack([noise_power] * 3, dim=2)

        # # render
        hf = lwr * lw_height
        # hf_bump = self.computeBumpHeight(fsigmax, fsigmay, fscale)
        # hf += hf_bump
        hf += self.noise[:self.n, :self.n] * 0.005
        normal = normals(hf, self.size / self.n)
        x_dot_h, y_dot_h = self.computeTangent(normal)


        roughx = ((1 - lwr) * ew_rough + roughx).clamp(0,1)
        roughy = ((1 - lwr) * ew_rough + roughy).clamp(0,1)
        geom, n_dot_h = self.computeGeomTerm(normal)
    
        specular = geom * self.brdf_aniso(n_dot_h, x_dot_h, y_dot_h, roughx**2, roughy**2, 0.04)
        specular = specular.unsqueeze(2).repeat(1,1,3)
        geom = geom.unsqueeze(2).repeat(1,1,3)
        diffuse = geom * color / np.pi      
        img = (diffuse + specular) * light  

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0).clamp(min=eps))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        if not self.isHDR: 
            img = img.clamp(0,1).pow(1/2.2)

        if self.save_tex == 'yes':
            self.albedo_tex = color
            self.normal_tex = normal
            self.rough_tex = rough

        return img
