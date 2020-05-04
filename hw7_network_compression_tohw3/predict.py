import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from dataset import ImgDataset
from dataset import testTransform
from dataset import readfile
from model_StudentNet import StudentNet

#filename
workspace_dir = sys.argv[1]
output_filename = sys.argv[2]
model_filename = sys.argv[3]

print("Reading data")
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

# testing configuration
model_best = StudentNet(base=16).cuda() 
model_best.load_state_dict(torch.load(model_filename))

# testing dataset
batch_size = 48
test_set = ImgDataset(test_x, transform=testTransform)
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