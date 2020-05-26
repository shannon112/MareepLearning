import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from utils import same_seeds
from autoencoder_model import conv_autoencoder


same_seeds(0)

input_filename = sys.argv[1] # ~/Downloads/dataset/testX.npy
model_filename = sys.argv[2] # ~/checkpoints/baseline.pth
testX = np.load(input_filename)
print(testX.shape)
testX = np.transpose(testX, (0,3,1,2))
print(testX.shape)

# make dataset
data = torch.tensor(testX, dtype=torch.float)
test_dataset = TensorDataset(data)
test_dataloader = DataLoader(test_dataset, batch_size=1)

# model
model = conv_autoencoder().cuda()
model = torch.load(model_filename, map_location='cuda')
criterion = nn.MSELoss() #

# eval
loss_list = []
output_list = []
model.eval()
for data in test_dataloader:
    img = data[0].cuda()
    _,output = model(img)
    loss = criterion(output, img)
    loss_list.append(loss.item())
    output_list.append(output[0].cpu().detach().numpy())
loss_list = np.array(loss_list)
idx_loss_list = np.argsort(loss_list)

# get top n min & max indexes
n_idx = 20
indexes = []
for i in range(n_idx):
    print("min", idx_loss_list[i], loss_list[idx_loss_list[i]])
    indexes.append(idx_loss_list[i])
for i in range(n_idx)[::-1]:
    print("max", idx_loss_list[-(i+1)], loss_list[idx_loss_list[-(i+1)]])
    indexes.append(idx_loss_list[-(i+1)])

# plot original pictures
fig = plt.figure(figsize=(10,4))#10,4
imgs = testX[indexes]
for i, img in enumerate(imgs):
    img = np.transpose(img, (1, 2, 0))
    img = (img+1)/2
    plt.subplot(2, n_idx*2, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# plot reconstruct pictures
recs = np.array(output_list)[indexes]
for i, img in enumerate(recs):
    img = np.transpose(img, (1, 2, 0))
    img = (img+1)/2
    plt.subplot(2, n_idx*2, n_idx*2+i+1, xticks=[], yticks=[])
    plt.imshow(img)

fig.suptitle("cnn autoencoder reconstruction rmse: {:.4f} {:.4f} {:.4f} {:.4f}".format(loss_list[idx_loss_list[0]],
                                            loss_list[idx_loss_list[1]],
                                            loss_list[idx_loss_list[-2]],
                                            loss_list[idx_loss_list[-1]]))
#plt.tight_layout()
plt.show()