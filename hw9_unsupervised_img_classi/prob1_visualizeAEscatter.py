import sys
import os
import numpy as np
import torch

from utils import same_seeds
from utils import plot_scatter
from utils import cal_acc

from model_baseline import AE
from dataset_baseline import Image_Dataset
from clustering_baseline import predict
from clustering_baseline import inference

from model_strong import AE
from dataset_strong import Image_Dataset
from clustering_strong import predict
from clustering_strong import inference

same_seeds(0)

model_filename = sys.argv[1] # ~/checkpoints/baseline.pth
input_filename2 = sys.argv[2] # ~/Downloads/dataset/valX.npy
input_filename3 = sys.argv[3] # ~/Downloads/dataset/valY.npy
valX = np.load(input_filename2)
valY = np.load(input_filename3)

model = AE().cuda()
model.load_state_dict(torch.load(model_filename))
model.eval()

latents = inference(valX, model)
pred_from_latent, emb_from_latent = predict(latents)
acc_latent = cal_acc(valY, pred_from_latent)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
plot_scatter(emb_from_latent, valY, savefig='p1_baseline.png')