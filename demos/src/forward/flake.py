from base import *
from lookup import *

class Flake(Material):
    def __init__(self, datadir, args, device):
        print('Initial Class forward::Flake()')
        super().__init__(args, device)
        self.save_tex = args.save_tex
        self.op = args.operation
        self.paraPr = th.tensor([[ 999,  300,   0,2000], # light
                                 [ 0.4,  0.3,   0,   1], # albedo_r
                                 [ 0.4,  0.3,   0,   1], # albedo_g
                                 [ 0.4,  0.3,   0,   1], # albedo_b
                                 [0.04,0.005,0.03,0.05], # topF0
                                 [0.1,  0.05,0.01, 0.2], # topRough
                                 [ 0.1,  0.1,   0,   1], # flakeF0_r
                                 [ 0.1,  0.1,   0,   1], # flakeF0_g
                                 [ 0.1,  0.1,   0,   1], # flakeF0_b
                                 [ 0.1,  0.2,0.01, 0.8], # flakeRough
                                 [ 0.3,  0.2,0.01, 0.8], # flakeNDF
                                 [0.99,  0.1, 0.1,   1], # scale
                                 [   0,  0.5,  -1,   1], # shiftx
                                 [   0,  0.5,  -1,   1], # shifty
                                 [  15,    3,   5,  20]  # iSigma
                                 ], dtype=th.float32, device=device)

        # self.paraPr = th.tensor([[ 999,  300,    0,2000], # light
        #                          [ 0.04,  0.1,   0,   1], # albedo_r
        #                          [ 0.04,  0.1,   0,   1], # albedo_g
        #                          [ 0.04,  0.1,   0,   1], # albedo_b
        #                          [0.04,0.005,0.03,0.05], # topF0
        #                          [0.1,  0.05,0.01, 0.2], # topRough
        #                          [ 0.1,  0.1,   0,   1], # flakeF0_r
        #                          [ 0.1,  0.1,   0,   1], # flakeF0_g
        #                          [ 0.1,  0.1,   0,   1], # flakeF0_b
        #                          [ 0.2,  0.2,0.01, 0.8], # flakeRough
        #                          [ 0.2,  0.2,0.01, 0.8], # flakeNDF
        #                          [ 0.9,  0.1, 0.1,   1], # scale
        #                          [   0,  0.5,  -1,   1], # shiftx
        #                          [   0,  0.5,  -1,   1], # shifty
        #                          [  10,    3,   5,  20]  # iSigma
        #                          ], dtype=th.float32, device=device)

        self.paraCh = [1,3,1,1,3,1,1,1,2,1]
        self.paraNm = ['light','albedo','topF0','topRough','flakeF0','flakeRough','flakeNDF','scale','shift','iSigma']

        # self.flakeImgSize = 1024
        # self.generateFlakes(datadir, 0.003)
        # self.initFlakeNormal()
        self.loadFlakeNormal(datadir + 'flakes_0.001.png')

    def generateFlakes(self, datadir, _r):
        seed = np.random.randint(10000)
        fn_out = datadir + 'pdsample_%.4f_%d.out' % (_r, seed)
        cmd = datadir + 'PDSample/PDSample -r %f -s %d -o %s' % (_r, seed, fn_out)
        print(cmd)
        os.system(cmd)

        with open(fn_out, 'rb') as f:
            k = np.fromstring(f.read(4), np.int32, 1)[0]
            # print(k)
            radius = np.fromstring(f.read(4), np.float32, 1)[0]
            # print(radius)
            P = np.fromstring(f.read(k*8), np.float32, k*2)

        P.shape = k, 2

        # map to pixel units
        r = _r
        real2pix = self.flakeImgSize / 2.0
        P += 1
        P *= real2pix
        r *= real2pix
        r = np.int32(np.ceil(r * 2.5))
        # print(r)

        x = np.arange(2*r + 1, dtype=np.float32) - r
        x, y = np.meshgrid(x, x)
        dist = np.sqrt(x**2 + y**2)

        # padded
        d = np.ones((self.flakeImgSize + 2*r, self.flakeImgSize + 2*r), dtype=np.float32) * self.flakeImgSize
        idx = np.zeros((self.flakeImgSize + 2*r, self.flakeImgSize + 2*r), dtype=np.int32)
        P += r

        # plt.plot(P[:,0], P[:,1], '.')
        # plt.show()
        # exit()

        def splat(ij, flake_id):
            i0, j0 = ij - r
            i1, j1 = ij + r + 1
            crop = d[i0 : i1, j0 : j1]
            mask = dist < crop
            crop[mask] = dist[mask]
            d[i0 : i1, j0 : j1] = crop
            crop = idx[i0 : i1, j0 : j1]
            crop[mask] = np.int32(flake_id)
            idx[i0 : i1, j0 : j1] = crop

        for i in range(k):
            pt = np.int32(P[i, :])
            splat(pt, i)

        d = d[r:-r, r:-r]
        self.idx = idx[r:-r, r:-r]
        self.k = k

    def initFlakeNormal(self):
        normal_list = np.zeros((self.k, 3), dtype=np.float32)
        isGGX = False
        if (isGGX): print("... GGX ...")
        else: print("... Beckmanm ...")
        for i in range(self.k):
            xi = np.random.rand()
            if isGGX:
                t = np.sqrt(xi/(1.0-xi))
            else:
                t = np.sqrt(-np.log(xi))
            phi = 2.0*np.pi*np.random.rand()
            normal_list[i,0] = t * np.cos(phi)
            normal_list[i,1] = t * np.sin(phi)
            normal_list[i,2] = 1.0

        _dxyz = np.zeros((self.flakeImgSize, self.flakeImgSize, 3), dtype=np.float32)
        for i in range(self.flakeImgSize):
            for j in range(self.flakeImgSize):
                _dxyz[i,j,:] = normal_list[self.idx[i,j],:]

        Image.fromarray(np.uint8((_dxyz+1)/2*255)).save('flakes.png')
        exit()
        self.normal_dxyz = th.tensor(_dxyz, dtype=th.float32, device=self.device)

    def loadFlakeNormal(self, fn):
        _dxyz = (gyPIL2Array(Image.open(fn)) * 2) -1
        self.normal_dxyz = th.tensor(_dxyz, dtype=th.float32, device=self.device)

    def computeFlakeNormal(self, normal0, sigma):
        normal = th.stack([normal0[:,:,0]*sigma**2, normal0[:,:,1]*sigma**2, normal0[:,:,2]], 2)
        normal = normal.div(normal.norm(2.0, 2, keepdim=True).clamp(min=eps))
        return normal

    def brdf_color(self, n_dot_h, alpha, f0):
        D = self.ggx_ndf(n_dot_h, alpha)
        return gyDstack(D / (4 * (n_dot_h**2).clamp(eps,1)), f0)

    def resizedCenterCrop(self, I0, s, t, n):
        n0 = I0.size()[0]
        if self.op == 'generate' or self.op == 'train':
            I0 = I0.roll(np.random.randint(n0),0)
            I0 = I0.roll(np.random.randint(n0),1)

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

        diffuse_bottom = gyDstack(self.geom_planar, albedo/np.pi)
        specular_top = self.geom_planar * self.brdf(self.n_dot_h_planar, rough_top.pow(2.0), f0_top)
        specular_top = specular_top.unsqueeze(2).repeat(1,1,3)
        geom_flake = geom_flake.unsqueeze(2).repeat(1,1,3)
        specular_flake = geom_flake * self.brdf_color(n_dot_h_flake, rough_flake.pow(2.0), f0_flake)
        img = (diffuse_bottom + specular_top + specular_flake) * light

        tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0))).exp().unsqueeze(2).repeat(1,1,3)

        img *= tmp

        img = img.clamp(0,1).pow(1/2.2)

        if self.save_tex == 'yes':
            self.albedo_tex = gyDstack(th.ones_like(self.geom_planar), albedo)
            self.normal_flake_tex = (normal_flake + 1) / 2
            self.rough_flake_tex = th.ones_like(self.geom_planar) * rough_flake**2
            self.f0_flake_tex = gyDstack(th.ones_like(self.geom_planar), f0_flake)
            self.rough_top_tex = th.ones_like(self.geom_planar) * rough_top**2


        return img
