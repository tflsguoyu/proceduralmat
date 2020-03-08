from PIL import Image
from gytools import *
from forward import Material
import lookup


def lookup1d(A, x):
	return lookup.lookup1d(A, x).squeeze()

def lookup2d(A, x, y):
	return lookup.lookup2d(A, th.stack((x, y))).squeeze()

def lookup3d(A, p):
	return lookup.lookup2d(A, p.permute(2,0,1)).squeeze()

def mk_plane(angle):
	x = angle.item() * np.pi / 180
	c, s = np.cos(x), np.sin(x)
	right = [1, 0, 0]
	up = [0, s, c]
	return right, up

def ggx_ndf(cos_h, alpha):
    c2 = cos_h ** 2
    t2 = (1 - c2) / c2
    a2 = alpha ** 2
    denom = np.pi * c2**2 * (a2 + t2)**2
    return a2 / denom

def brdf(n_dot_h, alpha, f0):
    D = ggx_ndf(n_dot_h, alpha)
    return f0 * D / (4 * n_dot_h**2)


# all units in cm
class Wood(Material):
	def __init__(self, n=400, datadir='data', device='cpu'):
		super().__init__(n, 10.0, 20.0, True, device)
		self.noise1d = th.rand(100, 1)
		self.noise2d = th.rand(100, 100, 1)
		self.noise3d = th.rand(20, 20, 3)
		self.noise = from16bit(Image.open(datadir + '/bnw-spots.png'))

		self.params = [
			('color',         3, 0, 0.1, -1, 1),
			('center',        3, 0, 10, -50, 50),
			('ring_size',     1, 0.5, 0.2, 0.1, 1),
			('lw_power',      1, 2.5, 1, 1.2, 5),         # latewood power
			('lw_fraction',   1, 0.5, 0.2, 0.1, 0.9),     # latewood fraction
			('ew_ramp_width', 1, 0.2, 0.05, 0.01, 0.4),   # earlywood ramp
			('lw_ramp_width', 1, 0.2, 0.05, 0.01, 0.4),   # latewood ramp
			('ssn_scale',     1, 10, 5, 1, 50),           # small-scale noise scale
			('ssn_power',     1, 0.5, 0.2, 0, 2),         # small-scale noise power
			('grn_scale',     1, 0.5, 0.3, 0, 2),         # growth rate noise scale
			('grn_amplitude', 1, 0.5, 0.3, 0, 2),         # growth rate noise amplitude
			('gdn_scale',     1, 0.3, 0.2, 0, 1),         # global distortion noise scale
			('gdn_amplitude', 1, 0.5, 0.3, 0, 2),         # global distortion noise amplitude
			('rough',         1, 0.4, 0.2, 0.1, 0.8),
			('ew_rough',      1, 0.1, 0.1, 0, 0.3),       # earlywood roughness increase
			('lw_height',     1, 0.002, 0.002, 0, 0.01),  # latewood height increase
			('cut_angle',     1, 0, 10, -90, 90),
			('light',         1, 2000, 500, 0, 10000)
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

	def eval_render(self, x):
		color, center, ring_size, lw_power, lw_fraction, ew_ramp_width,\
			lw_ramp_width, ssn_scale, ssn_power, grn_scale, grn_amplitude,\
			gdn_scale, gdn_amplitude, rough, ew_rough, lw_height, cut_angle, light = self.unpack(x)
		color += th.tensor((0.6, 0.4, 0.1))
		color = color.clamp(0.01, 0.95)

		s = self.size / 2
		px = th.linspace(-s, s, self.n)
		py, px = th.meshgrid(px, px)

		X, Y = mk_plane(cut_angle)
		p = dstack(px, X) + dstack(py, Y) + dstack(th.ones_like(px), center)

		# apply deformation to p
		p += gdn_amplitude * lookup3d(self.noise3d, p * gdn_scale)

		x = p[:, :, 0]
		y = p[:, :, 1]
		r = th.sqrt(x**2 + y**2) # distance from tree center
		r += grn_amplitude * lookup1d(self.noise1d, r)

		# compute latewood ratio
		t = r.fmod(ring_size) / ring_size
		ramp1 = t / ew_ramp_width
		ramp2 = (lw_fraction - t) / lw_ramp_width + 1
		lwr = th.min(ramp1, ramp2).clamp(0, 1)

		# compute final color
		lw_color = color ** lw_power
		img = dstack(1-lwr, color) + dstack(lwr, lw_color)

		noise_power = lookup2d(self.noise2d, x * ssn_scale, y * ssn_scale) * ssn_power + 1
		color = img ** th.stack([noise_power] * 3, dim=2)

		# render
		hf = lwr * lw_height
		hf += self.noise[:self.n, :self.n] * 0.005
		normal = normals(hf, self.size / self.n)
		rough = ((1 - lwr) * ew_rough + rough).clamp(0,1)
		n_dot_h = (self.omega * normal).sum(2).clamp(0,1)
		geom = n_dot_h / self.dist_sq

		specular = geom * brdf(n_dot_h, rough**2, 0.04) * light
		specular = th.stack([specular] * 3, 2)
		diffuse = dstack(geom, [light / np.pi] * 3) * color
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
	model = Wood()
	imgs = []

	for i in range(64):
		x = model.sample_prior()
		img = model.eval_render(x)
		imgs.append(img.permute(2,0,1))

	from torchvision.utils import make_grid
	grid = make_grid(imgs, 8).permute(1,2,0)
	grid = grid.clamp(0,1).pow(1 / 2.2) * 255
	Image.fromarray(grid.numpy().astype(np.uint8)).save('wood.png')
