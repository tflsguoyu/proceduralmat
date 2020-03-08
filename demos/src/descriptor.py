from torchvision.models.vgg import vgg19
from torchvision.transforms import Normalize, Compose, ToTensor
import torch as th



class TextureDescriptor(th.nn.Module):

	def __init__(self, device):
		super(TextureDescriptor, self).__init__()
		self.device = device
		self.outputs = []

		# get VGG19 feature network in evaluation mode
		self.net = vgg19(True).features.to(device)
		self.net.eval()

		# change max pooling to average pooling
		for i, x in enumerate(self.net):
			if isinstance(x, th.nn.MaxPool2d):
				self.net[i] = th.nn.AvgPool2d(kernel_size=2)

		def hook(module, input, output):
			self.outputs.append(output)

		# print(self.net)

		# for i in [6, 13, 26, 39]: # with BN
		for i in [4, 9, 18, 27]: # without BN
			self.net[i].register_forward_hook(hook)

		# weight proportional to num. of feature channels [Aittala 2016]
		self.weights = [1, 2, 4, 8, 8]
		
		# this appears to be standard for the ImageNet models in torchvision.models;
		# takes image input in [0,1] and transforms to roughly zero mean and unit stddev
		self.normalize_xform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.full_xform = Compose([ToTensor(), self.normalize_xform])


	def forward(self, x):
		self.outputs = []

		# run VGG features
		x = self.net(x.unsqueeze(0))
		self.outputs.append(x)
		
		result = []
		for i, F in enumerate(self.outputs):
			F = F.squeeze()
			f, s1, s2 = F.shape
			s = s1 * s2
			F = F.view((f, s))

			# Gram matrix
			G = th.mm(F, F.t()) / s
			result.append(G.flatten()) # * self.weights[i])

		return th.cat(result)


	def eval_image_0_255(self, x):
		"this takes a PIL Image or numpy array of size H x W x 3, between [0,255], gamma space"
		x = self.full_xform(x)
		return self.forward(x.to(self.device))
