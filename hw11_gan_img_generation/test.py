import torch
from utils import same_seeds
from model import Generator

same_seeds(0)

workspace_dir = sys.argv[1] #~/Downloads/faces/
img_dir = sys.argv[2] #./img

# load pretrained model
G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(model_dir, 'dcgan_g.pth')))
G.eval()
G.cuda()

# generate images and save the result
n_output = 20
z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0
save_dir = os.path.join('./log')
filename = os.path.join(img_dir, f'result.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=10)
# show image
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()