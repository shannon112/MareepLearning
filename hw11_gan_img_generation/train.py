
import os
import sys
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from preprocess import FaceDataset
from preprocess import get_dataset
from utils import same_seeds
from model import Generator
from model import Discriminator
from model import weights_init

workspace_dir = sys.argv[1] #~/Downloads/faces/
model_dir = sys.argv[2] #./model
save_dir = os.path.join('./log')
os.makedirs(save_dir, exist_ok=True)
same_seeds(0)

# hyperparameters 
batch_size = 64
z_dim = 100
lr = 1e-4
n_epoch = 30

# model
G = Generator(in_dim=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()
# loss criterion
criterion = nn.BCELoss()
# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

# dataloader (You might need to edit the dataset path if you use extra dataset.)
dataset = get_dataset(os.path.join(workspace_dir))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# show one image
plt.imshow(dataset[10].numpy().transpose(1,2,0))
plt.show()

# for logging
z_sample = Variable(torch.randn(100, z_dim)).cuda()

# main training loop
for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()

        bs = imgs.size(0)

        """ Train D """
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)

        # label        
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()

        # dis
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        
        # compute loss
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = (r_loss + f_loss) / 2

        # update model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        """ train G """
        # leaf
        z = Variable(torch.randn(bs, z_dim)).cuda()
        f_imgs = G(z)

        # dis
        f_logit = D(f_imgs)
        
        # compute loss
        loss_G = criterion(f_logit, r_label)

        # update model
        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # log
        print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    G.eval()

    # save some sample
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    print(f' | Save some samples to {filename}.')

    # save model
    torch.save(G.state_dict(), os.path.join(model_dir, 'dcgan_g.pth'+str(e+1)))
    torch.save(D.state_dict(), os.path.join(model_dir, 'dcgan_d.pth'+str(e+1)))

    # show generated image
    #grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    #plt.figure(figsize=(10,10))
    #plt.imshow(grid_img.permute(1, 2, 0))
    #plt.show()

    G.train()
    
