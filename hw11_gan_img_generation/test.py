import sys
import os
import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

from utils import same_seeds
from model import Generator

same_seeds(0)

workspace_dir = sys.argv[1] #~/Downloads/faces/
img_dir = sys.argv[2] #./img
model_filename = sys.argv[3]
model_name = model_filename.strip().split("/")[-1]

z_dim = 100

# load pretrained model
G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(model_filename)))
G.eval()
G.cuda()

# generate images and save the result
n_output = 20

for i in range(40):
    torch.manual_seed(i)
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(img_dir, 'result_{}.jpg'.format(i))
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10,2))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title("{}, randn seed {}".format(model_name,i))
    plt.savefig(filename)
    #torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    plt.show()