
import os
import sys
import numpy as np
import torch

from dataset import MyDataset
from dataset import get_dataloader
from model_StudentNet_deeper import StudentNet
#from model_TeacherNet_lite import TeacherNet
from weight_quantization import decode8

#filename
workspace_dir = sys.argv[1]
output_filename = sys.argv[2]
model_filename = sys.argv[3]

# get dataloader
test_loader = get_dataloader(workspace_dir,'testing', batch_size=32)

# testing configuration
model_dict = decode8(model_filename)
model_best = StudentNet(base=16).cuda() 
#model_best = TeacherNet().cuda() 
model_best.load_state_dict(model_dict)

# predict
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        test_pred = model_best(inputs)
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

# Write the result to csv file
with open(output_filename, 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
print("saved into",output_filename)