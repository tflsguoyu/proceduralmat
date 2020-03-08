from PIL import Image
from gytools import *
from forward import Material


def ggx_ndf(cos_h, alpha):
    c2 = cos_h ** 2
    t2 = (1 - c2) / c2
    a2 = alpha ** 2
    denom = np.pi * c2**2 * (a2 + t2)**2
    return a2 / denom

def brdf(n_dot_h, alpha, f0):
    D = ggx_ndf(n_dot_h, alpha)
    return f0 * D / (4 * n_dot_h**2)



class Plaster(Material):
	def __init__(self, n=400, datadir='data', device='cpu'):
		super().__init__(n, 30.0, 30.0, None, None, True, device)
		self.noise = from16bit(Image.open(datadir + '/bnw-spots.png'))

		self.params = [
			('color', 3, 0.4, 0.3, 0, 1),
			('rough', 1, 0.4, 0.2, 0.1, 0.8),
			('rough_var', 1, 0.2, 0.1, 0, 0.4),
			('height', 1, 0.2, 0.1, 0, 0.5),
			('slope', 1, 2, 1, 1, 4),
			('light', 1, 2000, 500, 0, 10000)
		]

	def get_params(self):
		return self.params

	def unpack(self, x):
		i = 0
		u = []
		for p in self.params:
			d = p[1]
			u.append(x[i:i+d])
			i += d
		return u

	def eval_param_maps(self, rough, rough_var, noise_slope, height):
		n = self.n
		s = 0 # (1024 - n) // 2
		base_noise = self.noise[s:s+n, s:s+n]

		# compute heightfield and turn to normals
		hf = (base_noise * noise_slope).clamp(0, 1) * height
		normal = normals(hf, self.size / n)

		# roughness depends on base_noise: increase in lower areas
		rough = (1 - base_noise * 2) * rough_var + rough

		return normal, rough.clamp(0.1, 0.8)

	def eval_render(self, x):
		color, rough, rough_var, height, noise_slope, light = self.unpack(x)
		normal, rough = self.eval_param_maps(rough, rough_var, noise_slope, height)

		n_dot_h = (self.omega * normal).sum(2).clamp(0,1)
		geom = n_dot_h / self.dist_sq

		specular = geom * brdf(n_dot_h, rough**2, 0.04) * light
		specular = th.stack([specular] * 3, 2)
		diffuse = dstack(geom, light * color / np.pi)
		img = diffuse + specular

		if not self.isHDR: img = img.clamp(0,1)
		return img

	def eval_prior_lpdf(self, x):
		i = 0
		lpdf = 0.0

		for p in self.params:
			d = p[1]
			lpdf += normal_lpdf(x[i:i+d], p[2], p[3])
			i += d
		return u

	def sample_prior(self):
		xs = []
		for p in self.params:
			x = sample_normal(p[2], p[3], p[4], p[5], p[1])
			xs.append(x)
		return th.cat(xs)



if __name__ == '__main__':
	model = Plaster()
	imgs = []

	for i in range(64):
		x = model.sample_prior()
		print(i, x.cpu().numpy())
		img = model.eval_render(x)
		imgs.append(img.permute(2,0,1))

	from torchvision.utils import make_grid
	grid = make_grid(imgs, 8).permute(1,2,0)
	grid = grid.clamp(0,1).pow(1 / 2.2) * 255
	Image.fromarray(grid.numpy().astype(np.uint8)).save('plaster.png')
