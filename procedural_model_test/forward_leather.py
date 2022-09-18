from forward import * 

class Leather(Material):
    def __init__(self, imres, device):
        print('Initial Class forward::Leather()')
        super().__init__(imres, device)
        self.f0 = 0.04     
        self.paraPr = th.tensor([[999, 500,   0,2500], # light
                                 [0.4, 0.3,   0,   1], # albedo_r
                                 [0.4, 0.3,   0,   1], # albedo_g
                                 [0.4, 0.3,   0,   1], # albedo_b
                                 [0.2, 0.2, 0.1, 0.8], # rough
                                 [0.2, 0.1,   0, 0.4], # rough_var
                                 [0.2, 0.1,   0, 0.5], # height
                                 [1.5, 0.2,   1,   2], # power
                                 [0.2, 0.2,0.01,   1], # scale
                                 [  0, 0.5,  -1,   1], # shiftx
                                 [  0, 0.5,  -1,   1], # shifty
                                 [ 10,   3,   5,  20]  # iSigma
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,1,1,1,1,1,2,1]
        self.paraNm = ['light','albedo','rough','rough_var','height','power','scale','shift','iSigma']
        
        self.noise = from16bit(Image.open('data/bnw-spots.png')).to(device)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[0]),0)
        self.noise = self.noise.roll(np.random.randint(self.noise.size()[1]),1)
               
        self.cells = from16bit(Image.open('data/voronoi-edges.png')).to(device)
        self.cells = self.cells.roll(np.random.randint(self.cells.size()[0]),0)
        self.cells = self.cells.roll(np.random.randint(self.cells.size()[1]),1)
 
    def resizedCenterCrop(self, I0, s, t, n):
        n0 = I0.size()[0]

        v = th.arange(n, dtype=th.float32, device=self.device)
        i, j = th.meshgrid((v, v))
        P = th.stack([i, j], 0)
        
        t = t.unsqueeze(-1).unsqueeze(-1).expand_as(P)
        P0 = x2p((p2x(P,n)+t)*s, n0)

        return lookup2d(I0.unsqueeze(-1), P0).squeeze(-1)

    def eval_param_maps(self, rough, rough_var, height, power, scale, shift):
        sizePerPixel = float(self.size) / self.n
        
        base_cells = self.resizedCenterCrop(self.cells, scale, shift, self.n).clamp(eps,1-eps) 
        noise = self.resizedCenterCrop(self.noise, scale, shift, self.n)  

        # compute heightfield and turn to normals
        hf = (1-base_cells**power) * height + (noise*2 - 1) * 0.005
        normal = normals(hf, sizePerPixel)

        # roughness depends on base_noise: increase in lower areas
        rough = base_cells * rough_var + rough

        return normal, rough.clamp(0.1, 0.8)

    def eval_render(self, x=None):
        light, albedo, rough, rough_var, height, power, scale, shift, iSigma = self.unpack(x)
        
        normal_cell, rough = self.eval_param_maps(rough, rough_var, height, power, scale, shift)
        geom_cell, n_dot_h_cell = self.computeGeomTerm(normal_cell)

        diffuse = dstack(geom_cell, albedo/np.pi)
        specular = geom_cell * self.brdf(n_dot_h_cell, rough.pow(2.0), self.f0)
        specular = specular.unsqueeze(2).repeat(1,1,3)
        img = (diffuse + specular) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0).clamp(min=eps))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        return img
