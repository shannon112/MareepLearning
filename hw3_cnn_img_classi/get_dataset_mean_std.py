import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import sys

# model
from model_vgg16_lite import Classifier

# dataset
from dataset import ImgDataset
from dataset import test_transform

#filename
workspace_dir = sys.argv[1]

# read img, resize img and get label from filename
def readfile(path, label):    
    img_size = 128
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), img_size, img_size, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(img_size, img_size))
        if label:
          y[i] = int(file.split("_")[0])
    # if label=true: train&valid
    if label:
      return x, y
    # if label=false: test
    else:
      return x

# reading testing set
print("Reading data")
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
train_x = readfile(os.path.join(workspace_dir, "training"), False)
print("Size of training data = {}".format(len(train_x)))
val_x = readfile(os.path.join(workspace_dir, "validation"), False)
print("Size of validation data = {}".format(len(val_x)))

# concatenate all avaliable data
test_x = np.concatenate((train_x, val_x,test_x), axis=0)
print(test_x.shape) #(16643, 128, 128, 3)

# testing dataset
batch_size = 100
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# calculating
nimages = 0
mean = 0.
std = 0.
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        print(batch.shape) #torch.Size([100, 3, 112, 112])
        
        # Rearrange batch to be the shape of [B,C, W * H]
        batch = batch.view(batch.size(0),batch.size(1),-1)
        print(batch.shape) #torch.Size([100, 3, 12544])
        print(batch)

        # Update total number of images
        nimages += batch.size(0)

        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)

# Final step
mean /= nimages
std /= nimages
print(mean)
print(std)
