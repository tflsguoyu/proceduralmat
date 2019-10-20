from PIL import Image
import torch as th
import numpy as np
from sys import argv
from descriptor import *


def to8bit(x):
    x = x.permute(1,2,0).detach().cpu().numpy()
    x *= np.array([0.229, 0.224, 0.225])
    x += np.array([0.485, 0.456, 0.406])
    x *= 255
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
td = TextureDescriptor(device)

# freeze the weights of td
for p in td.parameters():
	p.requires_grad = False

# load image
img = Image.open(argv[1])

# use as target
target_td = td.eval_image_0_255(img)
target_td.requires_grad = False

# optimize
X = th.randn(3, 400, 400, device=device)
X = th.nn.Parameter(X)
X.requires_grad = True

loss = th.nn.MSELoss()
optimizer = th.optim.Adam([X], lr=0.01)

for epoch in range(50):
	for i in range(500):
		if X.grad is not None:
			X.grad.detach_()
			X.grad.zero_()

		err = loss(td(X), target_td)
		err.backward()
		optimizer.step()

		if i % 10 == 0: print(epoch, i, err.item())

	out = to8bit(X)
	out.save('r' + str(epoch) + '.png')
