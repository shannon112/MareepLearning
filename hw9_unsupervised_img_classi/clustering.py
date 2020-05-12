import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

from model_strong import AE
#from model_baseline import AE

from dataset import preprocess
from dataset import Image_Dataset
from dataset import test_transform

input_filename = sys.argv[1] # ~/Downloads/dataset/trainX.npy
model_filename = sys.argv[2]  # ./model
output_predir = sys.argv[3] # ./submission

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X,test_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction to 200
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('PCA for First Reduction Shape:', kpca.shape)

    # Second Dimesnion Reduction to 2
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('TSNE for Second Reduction Shape:', X_embedded.shape)

    # Clustering to 2 Class
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

# load model
model = AE().cuda()
model.load_state_dict(torch.load(os.path.join(model_filename)))
model.eval()

# load training data
trainX = np.load(input_filename)

# extract latent vector
latents = inference(X=trainX, model=model)

# two step dimesion reduction and clustering 
pred, X_embedded = predict(latents)

# binary classification result
save_prediction(pred, os.path.join(output_predir,'prediction.csv'))
save_prediction(invert(pred), os.path.join(output_predir,'inverse_prediction.csv'))