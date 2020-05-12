import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import same_seeds
from utils import cal_acc
from model_strong import AE
from dataset_strong import Image_Dataset
from dataset_strong import test_transform
from clustering_strong import predict
from clustering_strong import inference

same_seeds(0)

# load checkpoint name
input_filename = sys.argv[1] # ~/Downloads/dataset/trainX.npy
input_filename2 = sys.argv[2] # ~/Downloads/dataset/valX.npy
input_filename3 = sys.argv[3] # ~/Downloads/dataset/valY.npy
checkpoints_list = sorted(glob.glob('checkpoints/strong_model/checkpoint_*.pth'))
print(checkpoints_list)

# load data
trainX = np.load(input_filename)
valX = np.load(input_filename2)
valY = np.load(input_filename3)
dataset = Image_Dataset(trainX,test_transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

# model
model = AE().cuda()
criterion = nn.MSELoss()

epochs = []
accs = []
losss = []
with torch.no_grad():
    for i, checkpoint in enumerate(checkpoints_list):
        # parsing checkpoint
        epoch = int(checkpoint.split("_")[-1].split(".")[0])

        # load checkpoint
        print('[{}/{}] {}'.format(i+1, len(checkpoints_list), checkpoint))
        model.load_state_dict(torch.load(checkpoint))
        model.eval()

        # preproduce checkpoint on trainX to get loss
        err = 0
        n = 0
        for i, x in enumerate(dataloader):
            img = x.cuda()
            img_encoded, img_decoded = model(img)
            loss = criterion(img_decoded, img)

            # Reconstruction error (MSE)
            err += loss.item()
            n += 1
        print('Reconstruction error (MSE):', err/n)

        # preproduce checkpoint on valX to get acc
        latents = inference(X=valX, model=model)
        pred, X_embedded = predict(latents)
        acc = cal_acc(valY, pred)
        print('Accuracy:', acc)
        
        # insert points
        epochs.append(epoch)
        losss.append(err/n)
        accs.append(acc)

# sorting
sort_idx = np.argsort(np.array(epochs))
losss = np.array(losss)[sort_idx].tolist()
accs = np.array(accs)[sort_idx].tolist()
epochs = np.array(epochs)[sort_idx].tolist()

# plot result
plt.figure(figsize=(6,6))
plt.subplot(211, title='Reconstruction error (MSE)').plot(epochs,losss)
plt.subplot(212, title='Accuracy (val)').plot(epochs,accs)
plt.show()
