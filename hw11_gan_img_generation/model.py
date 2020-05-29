import torch.nn as nn
import torch.nn.functional as F
from utils import same_seeds

same_seeds(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    convT map = (input-1) * S + opad -2*pad + ks
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False), #1 8192
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4), #2 8x8x256
            dconv_bn_relu(dim * 4, dim * 2), #3 16x16x128
            dconv_bn_relu(dim * 2, dim),     #4 32x32x64
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1), #5 64x64x3
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x): # x = 100
        y = self.l1(x)  # y = 8192
        y = y.view(y.size(0), -1, 4, 4) # y = 512x4x4
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    conv map = floor(W + 2*pad - ks)/S +1
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),#1 32x32x64
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),    #2 16x16x128
            conv_bn_lrelu(dim * 2, dim * 4),#3 8x8x256
            conv_bn_lrelu(dim * 4, dim * 8),#4 4x4x512
            nn.Conv2d(dim * 8, 1, 4),       #5 1x1x1
            nn.Sigmoid())
        self.apply(weights_init)        
    def forward(self, x): # x = 64x64x3
        y = self.ls(x)
        y = y.view(-1)
        return y

class Discriminator_noSigmoid(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    conv map = floor(W + 2*pad - ks)/S +1
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator_noSigmoid, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),#1 32x32x64
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),    #2 16x16x128
            conv_bn_lrelu(dim * 2, dim * 4),#3 8x8x256
            conv_bn_lrelu(dim * 4, dim * 8),#4 4x4x512
            nn.Conv2d(dim * 8, 1, 4),       #5 1x1x1
        )
        self.apply(weights_init)        
    def forward(self, x): # x = 64x64x3
        y = self.ls(x)
        y = y.view(-1)
        return y