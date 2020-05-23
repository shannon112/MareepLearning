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
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


test_filename = sys.argv[1]
model_filename = sys.argv[2]
batch_size = 128
same_seeds(0)

# load data
test = np.load(test_filename, allow_pickle=True)
print("data", test.shape)
# input shape
y = test

# make dataset
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

# load model
model = torch.load(model_filename, map_location='cuda')
model.eval()

# extract latent vector
latents = []
for i, data in enumerate(test_dataloader): 
    img = data[0].transpose(3, 1).cuda()
    vec, output = model(img)
    if i == 0: latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
    else: latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)

# Method 1: Clustering to 20 Class
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
    plt.savefig("img/cnn_clustered_tsne_result_{}.png".format(n+1))
    #plt.show()

    # output result
    with open('submission/prediction_cnn_{}.csv'.format(n+1), 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))

"""
# Method 2: PCA to 2D
pca = PCA(n_components=2).fit(latents)
y_projected = pca.transform(latents)
# judge by distribution
y_KMeans = MiniBatchKMeans(n_clusters=1, random_state=0).fit(y_projected)
y_cluster = y_KMeans.predict(y_projected)
dist = np.sum(np.square(y_KMeans.cluster_centers_[0] - y_projected), axis=1)
y_pred = dist
print(y_KMeans.cluster_centers_[0])
# judge by reconstruction error
#y_reconstructed = pca.inverse_transform(y_projected)  
#dist = np.sqrt(np.sum(np.square(y_reconstructed - latents).reshape(len(y), -1), axis=1))
#y_pred = dist
# plot clustering result
X = y_projected[:, 0]
Y = y_projected[:, 1]
plt.scatter(X, Y, s=1,c=y_cluster)
plt.legend()
plt.title("pca")
plt.savefig("img/cnn_pca_result.png")
# output result
with open('submission/prediction_cnn_pca.csv', 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
# score = roc_auc_score(y_label, y_pred, average='micro')
# score = f1_score(y_label, y_pred, average='micro')
# print('auc score: {}'.format(score))
"""