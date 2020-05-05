import os
import sys
import numpy as np
import torch

from dataset import MyDataset
from dataset import get_dataloader
from model_StudentNet import StudentNet
from weight_quantization import decode8

#filename
workspace_dir = sys.argv[1]
input_filename = sys.argv[2]

# get dataloader
test_loader = get_dataloader(workspace_dir,'validation', batch_size=1)

# predict
ref = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        for label in labels:
            ref.append(int(label.item()))

prediction = []
# Write the result to csv file
with open(input_filename, 'r') as f:
    f.readline()
    for i, line in  enumerate(f.readlines()):
        label = line.split(',')[1]
        prediction.append(int(label))

print(len(prediction),len(ref))
print(np.sum(np.array(prediction)==np.array(ref)))
print(np.sum(np.array(prediction)==np.array(ref))/len(prediction))