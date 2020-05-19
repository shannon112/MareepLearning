import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from autoencoder_model import fcn_autoencoder
from autoencoder_model import conv_autoencoder
from autoencoder_model import conv_autoencoder_hw9
from autoencoder_model import VAE
from autoencoder_model import loss_vae
from utils import same_seeds

same_seeds(0)
train_filename = sys.argv[1]
model_filename = sys.argv[2]
train = np.load(train_filename, allow_pickle=True)

# parameters
num_epochs = 200
batch_size = 128
learning_rate = 1e-3
model_type = model_filename.split('.')[-2][-3:] #{'fcn', 'cnn', 'vae'} 
model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder_hw9(), 'vae':VAE()}

# transform input X
x = train
if model_type == 'fcn' or model_type == 'vae':
    x = x.reshape(len(x), -1)

# make dataset
data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# model
model = model_classes[model_type].cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# main traning loop
best_loss = np.inf
model.train()
for epoch in range(num_epochs):
    for data in train_dataloader:
        # transform input X
        if model_type == 'cnn': img = data[0].transpose(3, 1).cuda()
        else: img = data[0].cuda()
        # ===================forward=====================
        if model_type == 'vae': output = model(img)      
        else: output = _,output = model(img)
        # loss
        if model_type == 'vae': loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else: loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================save====================
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model, model_filename)
            print("save")
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.item()))
