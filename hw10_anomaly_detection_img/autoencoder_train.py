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
from autoencoder_model import VAE
from autoencoder_model import loss_vae

train_filename = sys.argv[1]
test_filename = sys.argv[2]

train = np.load(train_filename, allow_pickle=True)
test = np.load(test_filename, allow_pickle=True)

num_epochs = 1000
batch_size = 128
learning_rate = 1e-3

#{'fcn', 'cnn', 'vae'} 
model_type = 'fcn' 

x = train
if model_type == 'fcn' or model_type == 'vae':
    x = x.reshape(len(x), -1)
    
data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE()}
model = model_classes[model_type].cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

best_loss = np.inf
model.train()
for epoch in range(num_epochs):
    for data in train_dataloader:
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        # ===================forward=====================
        output = model(img)
        if model_type == 'vae':
            loss = loss_vae(output[0], img, output[1], output[2], criterion)
        else:
            loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================save====================
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model, 'best_model_{}.pt'.format(model_type))
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.item()))
    

