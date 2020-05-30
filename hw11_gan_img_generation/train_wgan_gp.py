
import os
import sys
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import matplotlib.pyplot as plt

from preprocess import FaceDataset
from preprocess import get_dataset
from utils import same_seeds
from model import Generator
from model import Discriminator_noSigmoid

workspace_dir = sys.argv[1] #~/Downloads/faces/
model_dir = sys.argv[2] #./model
save_dir = os.path.join('./log')
os.makedirs(save_dir, exist_ok=True)
same_seeds(0)

# hyperparameters 
batch_size = 64
z_dim = 100
lr = 20*1e-4
n_epoch = 39
lambda_gp = 250
n_critic = 3

# model
G = Generator(in_dim=z_dim).cuda() #latent=100
#G.load_state_dict(torch.load('model/wgangp_g.pth30'))
print(G)
D = Discriminator_noSigmoid(in_dim=3).cuda() #channel=3
#D.load_state_dict(torch.load('model/wgangp_d.pth30'))
print(D)
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
#plt.imshow(dataset[10].numpy().transpose(1,2,0))
#plt.show()

# for logging
z_sample = Variable(torch.randn(100, z_dim)).cuda()

# main training loop
for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()
        bs = imgs.size(0)

        """ Train D """
        # sample real img from dataset
        r_imgs = Variable(imgs).cuda()
        # generator generate fake img from sample
        z = Variable(torch.randn(bs, z_dim)).cuda()
        f_imgs = G(z)

        # discriminator judging
        r_logit = D(r_imgs)
        f_logit = D(f_imgs)
        
        # gradient penalty
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((bs, 1, 1, 1))).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * r_imgs.detach() + ((1 - alpha) * f_imgs.detach())).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones((bs)).cuda()
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # compute loss
        loss_D = - torch.mean(r_logit) + torch.mean(f_logit) + lambda_gp * gradient_penalty
        #print(torch.mean(r_logit).item() , torch.mean(f_logit).item(), gradient_penalty.item())

        # update discriminator model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()
        G.zero_grad()

        """ train G """
        if i % n_critic == 0:
            # generator generate fake img from sample
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            # discriminator judging
            f_logit = D(f_imgs)
            
            # compute loss
            loss_G = -torch.mean(f_logit)

            # update generator model
            loss_G.backward()
            opt_G.step()

        # log
        print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    G.eval()

    # save some sample
    #f_imgs_sample = (G(z_sample).data + 1) / 2.0
    #filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
    #torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    #print(f' | Save some samples to {filename}.')

    # save several models
    #torch.save(G.state_dict(), os.path.join(model_dir, 'wgangp_g.pth'+str(e+1)))
    #torch.save(D.state_dict(), os.path.join(model_dir, 'wgangp_d.pth'+str(e+1)))

    # show generated image
    #grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    #plt.figure(figsize=(10,10))
    #plt.imshow(grid_img.permute(1, 2, 0))
    #plt.show()

    G.train()
    
torch.save(G.state_dict(), os.path.join(model_dir))
#torch.save(D.state_dict(), os.path.join(model_dir, 'wgangp_d.pth'+str(e+1)))
