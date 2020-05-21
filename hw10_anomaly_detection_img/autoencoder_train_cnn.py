import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import torchvision.transforms as transforms

from autoencoder_model import conv_autoencoder
from autoencoder_model import conv_autoencoder_hw9
from utils import same_seeds

same_seeds(0)
train_filename = sys.argv[1]
model_filename = sys.argv[2]
train = np.load(train_filename, allow_pickle=True)
print(train.shape)

# parameters
num_epochs = 200
batch_size = 128
learning_rate = 1e-3

# make dataset
data = torch.tensor(train, dtype=torch.float)
data = (data + 1) / 2
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# model
model = conv_autoencoder().cuda()
criterion = nn.BCELoss() #nn.MSELoss() #
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# main traning loop
best_loss = np.inf
model.train()
for epoch in range(num_epochs):
    for data in train_dataloader:
        # transform input X
        img = data[0].transpose(3, 1).cuda()
        # ===================forward=====================
        _,output = model(img)
        # loss
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================save====================
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model, model_filename+str(epoch))
        print("save")
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.item()))
