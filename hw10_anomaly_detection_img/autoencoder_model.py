import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import same_seeds

same_seeds(0)

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True), 
            nn.Linear(64, 16), 
            nn.ReLU(True), 
            nn.Linear(16, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(True),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True), 
            nn.Linear(256, 32 * 32 * 3), 
            nn.Tanh()
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        x = self.decoder(x_latent)
        return x_latent, x


class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2), #16
            nn.Conv2d(12, 24, 3, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2), #8
			nn.Conv2d(24, 48, 3, stride=1, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.MaxPool2d(2), #4
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 5, stride=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 9, stride=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 17, stride=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        x = self.decoder(x_latent)
        return x_latent, x


class conv_autoencoder_hw9(nn.Module):
    def __init__(self):
        super(conv_autoencoder_hw9, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        x = self.decoder(x_latent)
        return x_latent, x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(32*32*3, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 32*32*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return mse + KLD