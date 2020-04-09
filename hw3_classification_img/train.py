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
#from model_vgg16_lite_shallow import Classifier
#from model_dnn import Classifier

# dataset
from dataset import ImgDataset
from dataset import train_transform
from dataset import test_transform

#random.seed(args.seed)
torch.manual_seed(0)

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

# reading training set, validation set
workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

# train in training & validating set (final step)
#train_x = np.concatenate((train_x, val_x), axis=0)
#train_y = np.concatenate((train_y, val_y), axis=0)

# create train and valid dataset
batch_size = 48
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Keep the loss and accuracy at every iteration for plotting
train_loss_list = []
dev_loss_list = []
train_acc_list = []
dev_acc_list = []

# training configuration
model = Classifier().cuda()
#model_filename = "./model/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112/lr0001_train_n_val/model_0.9848396501457726_ep150"
#model.load_state_dict(torch.load(model_filename))
loss = nn.CrossEntropyLoss() # due to classification task，we use CrossEntropyLoss
#optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # optimizer use Adam
optimizer = torch.optim.SGD(model.parameters(), 0.0002, momentum=0.9, weight_decay=1e-4)
num_epoch = 50
val_acc_max = 0.0
train_acc_max = 0.0

# training loop
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # ensure models is at train model (enable Dropout, etc.)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda()) # using model get prediction, actually call model' forward function
        batch_loss = loss(train_pred, data[1].cuda()) # calculating loss ( prediction and label need to be on CPU or GPU togather）
        batch_loss.backward() # back propagation get every weight's gradient
        optimizer.step() # use gradient update weights

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

        train_loss_list.append(train_loss/train_set.__len__())
        dev_loss_list.append(val_loss/val_set.__len__())
        train_acc_list.append(train_acc/train_set.__len__())
        dev_acc_list.append(val_acc/val_set.__len__())

        if ((val_acc/val_set.__len__()) > val_acc_max) and (train_acc/train_set.__len__()>train_acc_max):
            val_acc_max = val_acc/val_set.__len__()
            train_acc_max = train_acc/train_set.__len__()
            print("save")
            #torch.save(model.state_dict(), "./model_"+str(val_acc/val_set.__len__()))
#torch.save(model.state_dict(), "./model_last")

# plotting result
import matplotlib.pyplot as plt

# Loss curve
plt.plot(train_loss_list)
plt.plot(dev_loss_list)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc_list)
plt.plot(dev_acc_list)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()
