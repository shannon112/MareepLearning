import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils import same_seeds

test_filename = sys.argv[1]
model_filename = sys.argv[2]
batch_size = 128
same_seeds(0)

# load data
test = np.load(test_filename, allow_pickle=True)
model_type = model_filename.split('.')[-2][-3:]

# input shape
if model_type == 'fcn' or model_type == 'vae': y = test.reshape(len(test), -1)
else: y = test

# make dataset
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

# load model
model = torch.load('best_model_{}.pt'.format(model_type), map_location='cuda')
model.eval()

# reconstruct img
reconstructed = []
for i, data in enumerate(test_dataloader): 
    if model_type == 'cnn':
        img = data[0].transpose(3, 1).cuda()
    else:
        img = data[0].cuda()
        
    if model_type == 'vae': output = model(img)
    else: _, output = model(img)

    if model_type == 'cnn':
        output = output.transpose(3, 1)
    elif model_type == 'vae':
        output = output[0]
    reconstructed.append(output.cpu().detach().numpy())

# compute anomality
reconstructed = np.concatenate(reconstructed, axis=0)
anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
y_pred = anomality

# outout result
with open('prediction.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
# score = roc_auc_score(y_label, y_pred, average='micro')
# score = f1_score(y_label, y_pred, average='micro')
# print('auc score: {}'.format(score))
