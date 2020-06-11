from base import *

class Metal(Material):
    def __init__(self, args, device):
        print('Initial Class forward::Metal()')
        super().__init__(args, device)
        self.save_tex = args.save_tex
        # ###synthetic
        self.paraPr = th.tensor([[1500, 300, 100,2500], # light
                                 [0.4, 0.1, 0,   0.2], # f0_r
                                 [0.4, 0.1, 0,   0.2], # f0_g
                                 [0.4, 0.1, 0,   0.2], # f0_b
                                 [0.3, 0.1, 0.2,  0.8], # rough_x
                                 [0.3, 0.1, 0.2,  0.8], # rough_y
                                 [0.01, 2, 0,   1], # fsigma_x
                                 [10,  3, 1,   500], # fsigma_y
                                 [0.01,0.1, 0, 0.1], # fscale
                                 [ 15,   3, 5,   20]  # iSigma
                                 ], dtype=th.float32, device=device)
        # self.paraPr = th.tensor([[999, 300, 0,2000], # light
        #                          [0.4, 0.2, 0,   1], # f0_r
        #                          [0.4, 0.2, 0,   1], # f0_g
        #                          [0.4, 0.2, 0,   1], # f0_b
        #                          [0.2, 0.1, 0.01,  0.8], # rough_x
        #                          [0.2, 0.1, 0.01,  0.8], # rough_y
        #                          [0.01, 1, 0.01, 20], # fsigma_x
        #                          [ 20,  1, 0.01,  20], # fsigma_y
        #                          [0.01,0.02, 0, 0.1], # fscale
        #                          [ 10,   3, 5,   20]  # iSigma
        #                          ], dtype=th.float32, device=device)
        self.paraCh = [1,3,1,1,1,1,1,1]
        self.paraNm = ['light','f0','roughx','roughy','fsigmax','fsigmay','fscale','iSigma']

    def initPhase(self):
        self.sizePerPixel = float(self.size) / self.n
        self.phase = 2 * np.pi * th.rand(self.n, self.n, device=self.device)

        vF = th.arange(self.n, dtype=th.float32, device=self.device)
        vF = ((vF + 0.5) / self.n - 0.5) / self.sizePerPixel
        self.yF, self.xF = th.meshgrid((vF, vF))

    def ggx_ndf_aniso(self, cos_h, cos_hx, cos_hy, alpha_x, alpha_y):
        denom = np.pi * alpha_x * alpha_y * \
            (cos_hx**2/(alpha_x**2).clamp(0,1) + cos_hy**2/(alpha_y**2).clamp(0,1) + cos_h**2)**2
        return 1.0 / denom.clamp(min=eps)

    def brdf_aniso(self, n_dot_h, x_dot_h, y_dot_h, alpha_x, alpha_y, f0):
        D = self.ggx_ndf_aniso(n_dot_h, x_dot_h, y_dot_h, alpha_x, alpha_y)
        return gyDstack(D / (4 * (n_dot_h**2).clamp(eps,1)), f0)

    def computeBumpNormal(self, fsigmax, fsigmay, fscale):
        self.initPhase()
        amp = (-0.5 * ((self.xF/fsigmax.clamp(min=eps)).pow(2.0) + (-self.yF/fsigmay.clamp(min=eps)).pow(2.0))).exp()
        amp = gyShift(amp*fscale)
        profile = th.stack((amp*th.cos(self.phase), amp*th.sin(self.phase)), 2)
        hf = th.ifft(profile, 2)[:, :, 0] / (self.sizePerPixel**2)
        return gyHeight2Normal(hf, self.sizePerPixel)

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
        light, f0, roughx, roughy, fsigmax, fsigmay, fscale, iSigma = self.unpack(x)

        normal_bump = self.computeBumpNormal(fsigmax, fsigmay, fscale)
        geom_bump, n_dot_h_bump = self.computeGeomTerm(normal_bump)

        x_dot_h_bump, y_dot_h_bump = self.computeTangent(normal_bump)

        # diffuse = dstack(geom_bump, f0/np.pi)
        geom_bump = geom_bump.unsqueeze(2).repeat(1,1,3)
        specular = geom_bump * self.brdf_aniso(
            n_dot_h_bump, x_dot_h_bump, y_dot_h_bump, roughx.pow(2.0), roughy.pow(2.0), f0)
        img = specular * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        if self.save_tex == 'yes':
            self.normal_tex = (normal_bump + 1) / 2
            self.roughx_tex = th.ones_like(n_dot_h_bump) * roughx**2
            self.roughy_tex = th.ones_like(n_dot_h_bump) * roughy**2
            self.f0_tex = gyDstack(th.ones_like(n_dot_h_bump), f0)

        return img
