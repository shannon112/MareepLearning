import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from autoencoder_model import conv_autoencoder
from autoencoder_model import conv_autoencoder_hw9
from utils import same_seeds
from dataset_strong import Image_Dataset
from dataset_strong import train_transform

same_seeds(0)
train_filename = sys.argv[1]
model_filename = sys.argv[2]
trainX = np.load(train_filename, allow_pickle=True)
print(trainX.shape)

# parameters
num_epochs = 200
batch_size = 64
learning_rate = 1e-4

# make dataset
trainX = torch.tensor(trainX, dtype=torch.float) #-1~1
img_dataset = Image_Dataset(trainX, train_transform)
img_dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True)

# model
model = conv_autoencoder().cuda()
criterion = nn.MSELoss() #nn.MSELoss() #
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# main traning loop
best_loss = np.inf
model.train()
for epoch in range(num_epochs):
    for data in img_dataloader:
        data = data.cuda()
        _,output = model(data)
        loss = criterion(output, data)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model, model_filename+str(epoch))
        print("save")
    
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.item()))
