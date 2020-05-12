import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

from utils import same_seeds
from model_baseline import AE
from dataset_baseline import preprocess
from dataset_baseline import Image_Dataset

same_seeds(0)
def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
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

if __name__ == "__main__":
    input_filename = sys.argv[1] # ~/Downloads/dataset/trainX.npy
    input_modeldir = sys.argv[2]  # ./model
    output_predir = sys.argv[3] # ./submission

    # load model
    model = AE().cuda()
    model.load_state_dict(torch.load(os.path.join(input_modeldir)))
    model.eval()

    # extract latent vector
    trainX = np.load(input_filename)

    # two step dimesion reduction and clustering 
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    # binary classification result
    save_prediction(pred, os.path.join(output_predir,'prediction.csv'))
    save_prediction(invert(pred), os.path.join(output_predir,'inverse_prediction.csv'))