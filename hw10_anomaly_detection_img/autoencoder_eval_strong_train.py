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
from sklearn.decomposition import KernelPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


test_filename = sys.argv[1]
train_filename = sys.argv[2]
model_filename = sys.argv[3]
batch_size = 128
same_seeds(0)

# load data
train = np.load(train_filename, allow_pickle=True)
test = np.load(test_filename, allow_pickle=True)
model_type = model_filename.split('.')[-2][-3:]
print("data", test.shape)

# input shape
if model_type == 'fcn' or model_type == 'vae': y = test.reshape(len(test), -1)
else: y = test
if model_type == 'fcn' or model_type == 'vae': x = train.reshape(len(test), -1)
else: x = test

# make dataset
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# load model
model = torch.load(model_filename, map_location='cuda')
model.eval()

# extract latent vector
train_latents = []
for i, data in enumerate(test_dataloader): 
    # transform input img
    if model_type == 'cnn': img = data[0].transpose(3, 1).cuda()
    else: img = data[0].cuda()
    
    vec, output = model(img)

    # transform latent vector
    if model_type == 'cnn' or model_type == 'fcn': 
        if i == 0: train_latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else: train_latents = np.concatenate((train_latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)

test_latents = []
for i, data in enumerate(test_dataloader): 
    # transform input img
    if model_type == 'cnn': img = data[0].transpose(3, 1).cuda()
    else: img = data[0].cuda()
    
    vec, output = model(img)

    # transform latent vector
    if model_type == 'cnn' or model_type == 'fcn': 
        if i == 0: test_latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else: test_latents = np.concatenate((test_latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)


# project test to train's k-mean centers
pred = MiniBatchKMeans(n_clusters=18, random_state=0).fit(train_latents)
pred_cluster_train = pred.predict(train_latents)
pred_cluster_test = pred.predict(test_latents)
pred_dist = np.sum(np.square(pred.cluster_centers_[pred_cluster_test] - test_latents), axis=1)
y_pred = pred_dist

# output result
with open('submission/new_prediction_{}.csv'.format(18), 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))


# train tsne visualize
X_embedded = TSNE(n_components=2).fit_transform(train_latents)
print('TSNE for Second Reduction Shape:', X_embedded.shape)
X = X_embedded[:, 0]
Y = X_embedded[:, 1]
plt.scatter(X, Y, s=1, c=pred_cluster_train)
plt.savefig("img/train.png")
plt.show()

# train tsne visualize
X_embedded = TSNE(n_components=2).fit_transform(test_latents)
print('TSNE for Second Reduction Shape:', X_embedded.shape)
X = X_embedded[:, 0]
Y = X_embedded[:, 1]
plt.scatter(X, Y, s=1, c=pred_cluster_test)
plt.savefig("img/test.png")
plt.show()


"""
# Clustering to 20 Class
for n in range(20):
    pred = MiniBatchKMeans(n_clusters=n+1, random_state=0).fit(X_embedded)
    pred_cluster = pred.predict(X_embedded)
    pred_dist = np.sum(np.square(pred.cluster_centers_[pred_cluster] - X_embedded), axis=1)
    y_pred = pred_dist

    # plot clustering result
    plt.scatter(X, Y, c=pred_cluster, s=1)
    plt.legend()
    plt.title("n_clusters = {}".format(n+1))
    plt.savefig("img/clustered_tsne_result_{}.png".format(n+1))
    plt.show()

    # output result
    with open('submission/prediction_{}.csv'.format(n+1), 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

# score = roc_auc_score(y_label, y_pred, average='micro')
# score = f1_score(y_label, y_pred, average='micro')
# print('auc score: {}'.format(score))
"""