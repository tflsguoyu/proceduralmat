from forward import * 

class Plaster(Material):
    def __init__(self, imres, device):
        print('Initial Class forward::Plaster()')
        super().__init__(imres, device)
        self.f0 = 0.04  
        self.paraPr = th.tensor([[999, 300, 100,2500], # light
                                 [0.4, 0.3,   0,   1], # albedo_r
                                 [0.4, 0.3,   0,   1], # albedo_g
                                 [0.4, 0.3,   0,   1], # albedo_b
                                 [0.3, 0.2, 0.1, 0.8], # rough
                                 [0.2, 0.1,   0, 0.4], # rough_var
                                 [0.2, 0.1,   0, 0.5], # height
                                 [  2,   1,   1,   4], # slope
                                 [0.5, 0.3,0.01,   1], # scale
                                 [  0, 0.5,  -1,   1], # shiftx
                                 [  0, 0.5,  -1,   1], # shifty
                                 [ 10,   3,   5,  20]  # iSigma
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,1,1,1,1,1,2,1]
        self.paraNm = ['light','albedo','rough','rough_var','height','slope','scale','shift','iSigma']
        
        self.noise = from16bit(Image.open('data/bnw-spots.png')).to(device)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[0]),0)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[1]),1)
        
    def resizedCenterCrop(self, I0, s, t, n):
        n0 = I0.size()[0]

        v = th.arange(n, dtype=th.float32, device=self.device)
        i, j = th.meshgrid((v, v))
        P = th.stack([i, j], 0)
        t = t.unsqueeze(-1).unsqueeze(-1).expand_as(P)
        P0 = x2p((p2x(P,n)+t)*s, n0)
        return lookup2d(I0.unsqueeze(-1), P0).squeeze(-1)

    def eval_param_maps(self, rough, rough_var, noise_slope, height, scale, shift):
        sizePerPixel = float(self.size) / self.n

        base_noise = self.resizedCenterCrop(self.noise, scale, shift, self.n)

        # compute heightfield and turn to normals
        hf = (base_noise * noise_slope).clamp(0, 1) * height
        normal = normals(hf, sizePerPixel)

        # roughness depends on base_noise: increase in lower areas
        rough = (1 - base_noise * 2) * rough_var + rough

        return normal, rough.clamp(0.1, 0.8)

    def eval_render(self, x=None):
        light, albedo, rough, rough_var, height, noise_slope, scale, shift, iSigma = self.unpack(x)

        normal_noise, rough = self.eval_param_maps(rough, rough_var, noise_slope, height, scale, shift)
        geom_noise, n_dot_h_noise = self.computeGeomTerm(normal_noise)

        diffuse = dstack(geom_noise, albedo/np.pi)
        specular = geom_noise * self.brdf(n_dot_h_noise, rough.pow(2.0), self.f0)
        specular = specular.unsqueeze(2).repeat(1,1,3)
        img = (diffuse + specular) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        return img
        