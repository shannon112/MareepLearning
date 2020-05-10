from torch.utils.data import DataLoader
import numpy as np
import sys
import os
import torch.nn as nn

from utils import same_seeds
from utils import count_parameters
from dataset import Image_Dataset
from dataset import preprocess
from model import AE

input_filename = sys.argv[1] # ~/Downloads/dataset/trainX.npy
output_modeldir = sys.argv[2] # ./model

trainX = np.load(sys.argv[1])
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)


import torch
from torch import optim

same_seeds(0)

model = AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()
n_epoch = 200

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)


# 主要的訓練過程
loss_min = 1
best_model = None
for epoch in range(n_epoch):
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(output_modeldir,'checkpoint_{}.pth'.format(epoch+1)))
    if loss.data < loss_min:
        best_model = model.state_dict()
        loss_min = loss.data
        print("save best")
            
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, loss.data))

# 訓練完成後儲存 model
torch.save(best_model, os.path.join(output_modeldir,'last_checkpoint.pth'))