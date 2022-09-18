from forward import *

class Flake(Material):
    def __init__(self, imres, device):
        print('Initial Class forward::Flake()')
        super().__init__(imres, device)
        self.paraPr = th.tensor([[ 150,  50,  10, 300], # light
                                 [ 0.4,  0.3,   0,   1], # albedo_r
                                 [ 0.4,  0.3,   0,   1], # albedo_g
                                 [ 0.4,  0.3,   0,   1], # albedo_b
                                 [0.04,0.005,0.03,0.05], # topF0
                                 [0.1,  0.05,0.01, 0.2], # topRough
                                 [ 0.3,  0.1,   0,   1], # flakeF0_r
                                 [ 0.3,  0.1,   0,   1], # flakeF0_g
                                 [ 0.3,  0.1,   0,   1], # flakeF0_b
                                 [ 0.3,  0.2,0.01, 0.8], # flakeRough
                                 [ 0.3,  0.2,0.01, 0.8], # flakeNDF
                                 [ 0.8,  0.2, 0.1,   1], # scale
                                 [   0,  0.5,  -1,   1], # shiftx
                                 [   0,  0.5,  -1,   1], # shifty
                                 [  10,    3,   5,  20]  # iSigma
                                 ], dtype=th.float32, device=device)
        self.paraCh = [1,3,1,1,3,1,1,1,2,1]
        self.paraNm = ['light','albedo','topF0','topRough','flakeF0','flakeRough','flakeNDF','scale','shift','iSigma']
        
        self.normal_dxyz = from8bit(Image.open('data/flake-normal.png')).to(device)
        self.normal_dxyz = self.normal_dxyz *2 -1 
        self.normal_dxyz = self.normal_dxyz.roll(np.random.randint(self.normal_dxyz.size()[0]),0)
        self.normal_dxyz = self.normal_dxyz.roll(np.random.randint(self.normal_dxyz.size()[1]),1)  

    def computeFlakeNormal(self, normal0, sigma):
        normal = th.stack([normal0[:,:,0]*sigma**2, normal0[:,:,1]*sigma**2, normal0[:,:,2]], 2)
        normal = normal.div(normal.norm(2.0, 2, keepdim=True).clamp(min=eps))    
        return normal

    def brdf_color(self, n_dot_h, alpha, f0):
        D = self.ggx_ndf(n_dot_h, alpha)
        return dstack(D / (4 * (n_dot_h**2).clamp(eps,1)), f0)

    def resizedCenterCrop(self, I0, s, t, n):
        n0 = I0.size()[0]        

        v = th.arange(n, dtype=th.float32, device=self.device)
        i, j = th.meshgrid((v, v))
        P = th.stack([i, j], 0)
        t = t.unsqueeze(-1).unsqueeze(-1).expand_as(P)
        P0 = x2p((p2x(P,n)+t)*s, n0)
        return lookup2d(I0, P0)
        
    def eval_render(self, X=None):
        light, albedo, f0_top, rough_top, f0_flake, rough_flake, sigma_ndf, scale, shift, iSigma = self.unpack(X) 

        normal_flake_orig = self.resizedCenterCrop(self.normal_dxyz, scale, shift, self.n)
        normal_flake = self.computeFlakeNormal(normal_flake_orig, sigma_ndf)
        geom_flake, n_dot_h_flake = self.computeGeomTerm(normal_flake)

        diffuse_bottom = dstack(self.geom_planar, albedo/np.pi)
        specular_top = self.geom_planar * self.brdf(self.n_dot_h_planar, rough_top.pow(2.0), f0_top)
        specular_top = specular_top.unsqueeze(2).repeat(1,1,3)
        geom_flake = geom_flake.unsqueeze(2).repeat(1,1,3)
        specular_flake = geom_flake * self.brdf_color(n_dot_h_flake, rough_flake.pow(2.0), f0_flake)
        img = (diffuse_bottom + specular_top + specular_flake) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0))).exp().unsqueeze(2).repeat(1,1,3)
        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        return img
