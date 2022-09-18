from forward import * 
    
class Bump(Material):
    def __init__(self, imres, device):
        print('Initial Class forward::Bump()')
        super().__init__(imres, device)
        self.f0 = 0.04     
        self.paraPr = th.tensor([[999, 400,   0,2500], # light
                                 [0.4, 0.3,   0,   1], # albedo_r
                                 [0.4, 0.3,   0,   1], # albedo_g
                                 [0.4, 0.3,   0,   1], # albedo_b
                                 [0.2, 0.1,0.01, 0.5], # rough
                                 [  1, 0.5,   0,   2], # fsigma
                                 [0.5, 0.2,   0,   1], # fscale
                                 [ 10,   3,   5,  20]  # iSigma
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,1,1,1,1]
        self.paraNm = ['light','albedo','rough','fsigma','fscale','iSigma']
        self.initPhase()

    def initPhase(self):
        self.sizePerPixel = float(self.size) / self.n
        self.phase = 2 * np.pi * th.rand(self.n, self.n, device=self.device)

        vF = th.arange(self.n, dtype=th.float32, device=self.device)
        vF = ((vF + 0.5) / self.n - 0.5) / self.sizePerPixel
        self.yF, self.xF = th.meshgrid((vF, vF))

    def computeBumpNormal(self, fsigma, fscale):
        amp = (-0.5 * ((self.xF/fsigma).pow(2.0) + (-self.yF/fsigma).pow(2.0))).exp()
        amp = shift(amp*fscale)
        profile = th.stack((amp*th.cos(self.phase), amp*th.sin(self.phase)), 2)
        hf = th.ifft(profile, 2)[:, :, 0] / (self.sizePerPixel**2)
        return normals(hf, self.sizePerPixel)

    def eval_render(self, x=None):
        light, albedo, rough, fsigma, fscale, iSigma = self.unpack(x)

        normal_bump = self.computeBumpNormal(fsigma, fscale)
        geom_bump, n_dot_h_bump = self.computeGeomTerm(normal_bump)

        diffuse = dstack(geom_bump, albedo/np.pi)
        specular = geom_bump * self.brdf(n_dot_h_bump, rough.pow(2.0), self.f0)
        specular = specular.unsqueeze(2).repeat(1,1,3)
        img = (diffuse + specular) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        return img
        