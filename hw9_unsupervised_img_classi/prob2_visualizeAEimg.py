import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import same_seeds
from utils import plot_scatter
from utils import cal_acc

from model_baseline import AE
from dataset_baseline import Image_Dataset
from dataset_baseline import preprocess
from clustering_baseline import predict
from clustering_baseline import inference

same_seeds(0)

model_filename = sys.argv[1] # ~/checkpoints/baseline.pth
input_filename = sys.argv[2] # ~/Downloads/dataset/trainX.npy
trainX = np.load(input_filename)
trainX_preprocessed = preprocess(trainX)

# model
model = AE().cuda()
model.load_state_dict(torch.load(model_filename))
model.eval()

# plot original pictures
plt.figure(figsize=(10,4))
indexes = [1,2,3,6,7,9]
imgs = trainX[indexes,]
for i, img in enumerate(imgs):
    plt.subplot(2, 6, i+1, xticks=[], yticks=[])
    plt.imshow(img)

# plot reconstruct pictures
inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
latents, recs = model(inp)
recs = ((recs+1)/2 ).cpu().detach().numpy()
recs = recs.transpose(0, 2, 3, 1)
for i, img in enumerate(recs):
    plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
    plt.imshow(img)

plt.tight_layout()
plt.show()