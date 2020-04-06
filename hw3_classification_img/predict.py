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
from model_vgg16 import Classifier

# dataset
from dataset import ImgDataset
from dataset import test_transform

#filename
workspace_dir = sys.argv[1]
output_filename = sys.argv[2]
model_filename = sys.argv[3]

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

# testing configuration
model_best = Classifier().cuda()
model_best.load_state_dict(torch.load(model_filename))
loss = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)

# testing dataset
batch_size = 48
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# predict
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# Write the result to csv file
with open(output_filename, 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
print("saved into",output_filename)