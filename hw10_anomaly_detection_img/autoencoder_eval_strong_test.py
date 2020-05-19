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
model_filename = sys.argv[2]
batch_size = 128
same_seeds(0)

# load data
test = np.load(test_filename, allow_pickle=True)
model_type = model_filename.split('.')[-2][-3:]
print("data", test.shape)

# input shape
if model_type == 'fcn' or model_type == 'vae': y = test.reshape(len(test), -1)
else: y = test

# make dataset
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

# load model
model = torch.load(model_filename, map_location='cuda')
model.eval()

# extract latent vector
#reconstructed = []
latents = []
for i, data in enumerate(test_dataloader): 
    # transform input img
    if model_type == 'cnn': img = data[0].transpose(3, 1).cuda()
    else: img = data[0].cuda()
    
    vec, output = model(img)

    # transform latent vector
    if model_type == 'cnn' or model_type == 'fcn': 
        if i == 0: latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else: latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)

    # transform output img
    #if model_type == 'cnn': output = output.transpose(3, 1)
    #elif model_type == 'vae': output = output[0]
    #reconstructed.append(output.cpu().detach().numpy())
#reconstructed = np.concatenate(reconstructed, axis=0)


# First Dimension Reduction to 200
#transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
#kpca = transformer.fit_transform(latents)
#print('PCA for First Reduction Shape:', kpca.shape)

# Second Dimesnion Reduction to 2
#X_embedded = TSNE(n_components=2).fit_transform(latents)
#print('TSNE for Second Reduction Shape:', X_embedded.shape)
#X = X_embedded[:, 0]
#Y = X_embedded[:, 1]

# plot data distribution
#plt.scatter(X, Y, s=1)
#plt.savefig("img/raw_tsne_result.png")
#plt.show()

# Clustering to 20 Class
for n in range(20):
    pred = MiniBatchKMeans(n_clusters=n+1, random_state=0).fit(latents)
    pred_cluster = pred.predict(latents)
    pred_dist = np.sum(np.square(pred.cluster_centers_[pred_cluster] - latents), axis=1)
    y_pred = pred_dist

    # plot clustering result
    X_embedded = TSNE(n_components=2).fit_transform(latents)
    print('TSNE for Second Reduction Shape:', X_embedded.shape)
    X = X_embedded[:, 0]
    Y = X_embedded[:, 1]
    plt.scatter(X, Y, c=pred_cluster, s=1)
    plt.legend()
    plt.title("n_clusters = {}".format(n+1))
    plt.savefig("img/clustered_tsne_result_{}.png".format(n+1))
    #plt.show()

    # output result
    with open('submission/prediction_{}.csv'.format(n+1), 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

# score = roc_auc_score(y_label, y_pred, average='micro')
# score = f1_score(y_label, y_pred, average='micro')
# print('auc score: {}'.format(score))
