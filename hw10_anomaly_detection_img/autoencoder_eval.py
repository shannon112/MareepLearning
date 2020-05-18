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


train_filename = sys.argv[1]
test_filename = sys.argv[2]
batch_size = 128

train = np.load(train_filename, allow_pickle=True)
test = np.load(test_filename, allow_pickle=True)


model_type = 'fcn' 

if model_type == 'fcn' or model_type == 'vae':
    y = test.reshape(len(test), -1)
else:
    y = test
    
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model = torch.load('best_model_{}.pt'.format(model_type), map_location='cuda')

model.eval()
reconstructed = list()
for i, data in enumerate(test_dataloader): 
    if model_type == 'cnn':
        img = data[0].transpose(3, 1).cuda()
    else:
        img = data[0].cuda()
    output = model(img)
    if model_type == 'cnn':
        output = output.transpose(3, 1)
    elif model_type == 'vae':
        output = output[0]
    reconstructed.append(output.cpu().detach().numpy())

reconstructed = np.concatenate(reconstructed, axis=0)
anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
y_pred = anomality
with open('prediction.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
# score = roc_auc_score(y_label, y_pred, average='micro')
# score = f1_score(y_label, y_pred, average='micro')
# print('auc score: {}'.format(score))
