import torch as th
import numpy as np
from utility import *

def paraZip(light, albedo, rough, fsigma, fscale, iSigma):
	light     = normTo01(light,  0, 1000)
	albedo    = normTo01(albedo, 0,    1)
	rough     = normTo01(rough,  0,    1)
	fsigma    = normTo01(fsigma, 0,    5)
	fscale    = normTo01(fscale, 0,  0.2)
	iSigma    = normTo01(iSigma, 0,   10)

	return [light, albedo[0], albedo[1], albedo[2], rough, fsigma, fscale, iSigma]

def paraUnzip(para):
	light  = normTo01_inv(para[0],   0, 1000)
	albedo = normTo01_inv(para[1:4], 0,    1)
	rough  = normTo01_inv(para[4],   0,    1)
	fsigma = normTo01_inv(para[5],   0,    5)
	fscale = normTo01_inv(para[6],   0,  0.2)
	iSigma = normTo01_inv(para[7],   0,   10)

	return light, albedo, rough, fsigma, fscale, iSigma

class Mfb:
	def __init__(self, n, size, camera, f0, device):
		print('Initial Class forward::Mfb()')
		self.n = n
		self.f0 = f0
		self.device = device

		self.initGeometry(size, camera)
		self.initPhase()

	def initGeometry(self, size, camera):
		self.size = size
		self.camera = camera
		
		# surface positions
		v = th.arange(self.n, dtype=th.float32, device=self.device)
		v = ((v + 0.5) / self.n - 0.5) * self.size
		y, x = th.meshgrid((v, v))
		pos = th.stack((x, -y, th.zeros_like(x)), 2)
		self.pos_norm = pos.norm(2.0, 2)

		# directions (omega_in = omega_out = half)
		self.omega = th.tensor([0,0,self.camera], dtype=th.float32, device=self.device) - pos
		self.dist_sq = self.omega.pow(2.0).sum(2)
		d = self.dist_sq.sqrt()
		self.omega /= th.stack((d, d, d), 2)

		normal = th.stack([th.zeros_like(d),th.zeros_like(d),th.ones_like(d)], 2)
		n_dot_h = (self.omega*normal).sum(2)
		self.geom = n_dot_h/self.dist_sq

	def initPhase(self):	
		self.sizePerPixel = float(self.size) / self.n
		self.phase = 2 * np.pi * th.rand(self.n, self.n, device=self.device)

		vF = th.arange(self.n, dtype=th.float32, device=self.device)
		vF = ((vF + 0.5) / self.n - 0.5) / self.sizePerPixel
		self.yF, self.xF = th.meshgrid((vF, vF))
		
	def beckmann_ndf(self, cos_h, alpha):
		c2 = cos_h ** 2
		t2 = (1 - c2) / c2
		a2 = alpha ** 2
		return th.exp(-t2 / a2) / (np.pi * a2 * c2**2)

	def ggx_ndf(self, cos_h, alpha):
		c2 = cos_h ** 2
		t2 = (1 - c2) / c2
		a2 = alpha ** 2
		denom = np.pi * c2**2 * (a2 + t2)**2
		return a2 / denom

	# def schlick(cos, f0):
	# 	return f0 + (1 - f0) * (1 - cos)**5

	def brdf(self, n_dot_h, alpha, f0):
		D = self.ggx_ndf(n_dot_h, alpha)
		return f0 * D / (4 * n_dot_h**2)

	def noise(self, fsigma, fscale):
		amp = (-0.5 * ((self.xF/fsigma).pow(2.0) + (-self.yF/fsigma).pow(2.0))).exp()
		amp = shift(amp*fscale)	
		profile = th.stack((amp*th.cos(self.phase), amp*th.sin(self.phase)), 2)
		return th.ifft(profile, 2)[:, :, 0] / (self.sizePerPixel**2)

	def eval(self, para):

		self.para = th.tensor(para, dtype=th.float32, device=self.device, requires_grad=True)
		light, albedo, rough, fsigma, fscale, iSigma = paraUnzip(self.para)
		
		light = light.clamp(min=0)
		albedo = albedo.clamp(0,1)
		rough = rough.clamp(0,1)
		fsigma = fsigma.clamp(min=0)
		fscale = fscale.clamp(min=0)
		iSigma = iSigma.clamp(min=0)

		# normals
		hf = self.noise(fsigma, fscale)
		normal_bump = normals(hf, self.sizePerPixel)

		# geometry term
		n_dot_h_bump = (self.omega*normal_bump).sum(2)
		geom_bump = n_dot_h_bump / self.dist_sq
		
		# brdf
		diffuse = dstack(geom_bump, light * albedo / np.pi)
		specular = geom_bump * self.brdf(n_dot_h_bump, rough.pow(2.0), self.f0) * light
		specular = th.stack((specular, specular, specular), 2)

		tmp = ((-self.pos_norm.pow(2.0))/(2*iSigma.pow(2.0))).exp()

		img = (diffuse + specular) * th.stack([tmp, tmp, tmp], 2)
		
		return img
