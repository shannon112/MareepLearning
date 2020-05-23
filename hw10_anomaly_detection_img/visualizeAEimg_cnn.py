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
from dataset_strong import Image_Dataset
from dataset_strong import test_transform


same_seeds(0)

input_filename = sys.argv[1] # ~/Downloads/dataset/testX.npy
model_filename = sys.argv[2] # ~/checkpoints/baseline.pth
testX = np.load(input_filename)

# make dataset
testX = torch.tensor(testX, dtype=torch.float) #-1~1
img_dataset = Image_Dataset(testX, test_transform)
img_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=True)

# model
#model = conv_autoencoder().cuda()
model = torch.load(model_filename, map_location='cuda')
criterion = nn.MSELoss() #

# eval
loss_list = []
output_list = []
model.eval()
for data in img_dataloader:
    # transform input X
    img = data.cuda()
    # ===================forward=====================
    _,output = model(img)
    loss = criterion(output, img)
    loss_list.append(loss.item())
    output_list.append(output[0].cpu().detach().numpy())
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
    img = (img+1) / 2
    plt.subplot(2, 4, i+1, xticks=[], yticks=[])
    plt.imshow(img)

output_list = np.array(output_list)[indexes]
output_list = ((output_list+1)/2 )
# plot reconstruct pictures
for i, img in enumerate(output_list):
    img = np.transpose(img, (1, 2, 0))
    img = (img+1) / 2
    plt.subplot(2, 4, 4+i+1, xticks=[], yticks=[])
    plt.imshow(img)

plt.tight_layout()
plt.show()

with open('prediction.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(loss_list)):
        f.write('{},{}\n'.format(i+1, loss_list[i]))
