import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sn

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
model_filename = sys.argv[2]

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
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of Testing data = {}".format(len(val_x)))

# testing dataset
batch_size = 48
val_set = ImgDataset(val_x, val_y, test_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# testing configuration
model_best = Classifier().cuda()
model_best.load_state_dict(torch.load(model_filename))

# predict
model_best.eval()
val_y_hat = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        test_pred = model_best(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            val_y_hat.append(y)

stacked = torch.stack((torch.tensor(val_y.tolist()), torch.tensor(val_y_hat)),dim=1)
print( stacked.shape )
print(stacked)

# confusion matrix in tensor
cmt = torch.zeros(11,11, dtype=torch.int64)
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
print(cmt)

# confusion matrix by sklearn in numpy
cm = confusion_matrix(torch.tensor(val_y.tolist()), torch.tensor(val_y_hat), normalize='true')
print(cm)

# visualization
target_names = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp = disp.plot(cmap=plt.cm.jet, xticks_rotation="45")
plt.title("confusion matrix")
plt.savefig('confusion_matrix.png')
plt.show()