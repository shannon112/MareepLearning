import os
import sys
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import same_seeds
from dataset_baseline import Image_Dataset
from dataset_baseline import preprocess
from model_baseline import AE

same_seeds(0)
input_filename = sys.argv[1] # ~/Downloads/dataset/trainX.npy
output_modeldir = sys.argv[2] # ./model
#output_filename = sys.argv[2] # ./chekpoints/baseline.pth

# dataset
trainX = np.load(sys.argv[1])
print("trainX", trainX.shape)
trainX_preprocessed = preprocess(trainX)
print("trainX_preprocessed", trainX_preprocessed.shape)
img_dataset = Image_Dataset(trainX_preprocessed)
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

# model, loss, optimizer
model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1*1e-5, weight_decay=1e-5)

# training parameters
n_epoch = 200
loss_min = 1
best_model = None

# main training loop
model.train()
for epoch in range(n_epoch):
    for data in img_dataloader:
        img = data
        img = img.cuda()

        img_encoded, img_decoded = model(img)
        loss = criterion(img_decoded, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save mid-model
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(output_modeldir,'checkpoint_{}.pth'.format(epoch+1)))
        pass
    if loss.data < loss_min:
        best_model = model.state_dict()
        loss_min = loss.data
        print("save best")           
        torch.save(model.state_dict(), os.path.join(output_modeldir,'checkpoint_{}.pth'.format(epoch+1)))
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

# save best model
torch.save(best_model, os.path.join(output_modeldir,'last_checkpoint.pth'))
#torch.save(best_model, os.path.join(output_filename))