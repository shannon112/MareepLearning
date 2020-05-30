import sys
import os
import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

from utils import same_seeds
from model import Generator

same_seeds(0)

model_filename = sys.argv[1]
img_filename = sys.argv[2] #./img
randn_seed = int(sys.argv[3])
model_name = model_filename.strip().split("/")[-1]

z_dim = 100

# load pretrained model
G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(model_filename)))
G.eval()
G.cuda()

# generate images and save the result
n_output = 20

torch.manual_seed(randn_seed)
z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
plt.figure(figsize=(10,2))
plt.imshow(grid_img.permute(1, 2, 0))
plt.title("{}, randn seed {}".format(model_name,randn_seed))
plt.savefig(img_filename)
#plt.show()