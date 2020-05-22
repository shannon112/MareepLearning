import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import torchvision.transforms as transforms

from utils import same_seeds
from autoencoder_model import conv_autoencoder


same_seeds(0)

input_filename = sys.argv[1] # ~/Downloads/dataset/testX.npy
model_filename = sys.argv[2] # ~/checkpoints/baseline.pth
testX = np.load(input_filename)

# make dataset
data = torch.tensor(testX, dtype=torch.float)
#data = (data + 1) / 2
test_dataset = TensorDataset(data)
test_sampler = RandomSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

# model
model = conv_autoencoder().cuda()
criterion = nn.MSELoss() #

# eval
loss_list = []
output_list = []
model.eval()
for data in test_dataloader:
    # transform input X
    img = data[0].transpose(3, 1).cuda()
    # ===================forward=====================
    _,output = model(img)
    loss = criterion(output, img)
    loss_list.append(loss.item())
    output_list.append(output.cpu().detach().squeeze().numpy())
loss_list = np.array(loss_list)
idx_loss_list = np.argsort(loss_list)
print("min", idx_loss_list[0], loss_list[idx_loss_list[0]])
print("min", idx_loss_list[1], loss_list[idx_loss_list[1]])
print("max", idx_loss_list[-2], loss_list[idx_loss_list[-2]])
print("max", idx_loss_list[-1], loss_list[idx_loss_list[-1]])
indexes = [idx_loss_list[0], idx_loss_list[1], idx_loss_list[-2], idx_loss_list[-1]]


# plot original pictures
plt.figure(figsize=(10,4))
imgs = testX[indexes]
for i, img in enumerate(imgs):
    plt.subplot(2, 4, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# plot reconstruct pictures
recs = np.array(output_list)[indexes]
recs = recs.transpose(0, 2, 3, 1)
print(recs.shape)
print(recs)
for i, img in enumerate(recs):
    plt.subplot(2, 4, 4+i+1, xticks=[], yticks=[])
    plt.imshow(img)

plt.tight_layout()
plt.show()