import sys
import numpy as np
from sklearn.decomposition import PCA

train_filename = sys.argv[1]
test_filename = sys.argv[2]

train = np.load(train_filename, allow_pickle=True)
test = np.load(test_filename, allow_pickle=True)

x = train.reshape(len(train), -1)
y = test.reshape(len(test), -1)
pca = PCA(n_components=2).fit(x)

y_projected = pca.transform(y)
y_reconstructed = pca.inverse_transform(y_projected)  
dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))

y_pred = dist
# score = roc_auc_score(y_label, y_pred, average='micro')
# score = f1_score(y_label, y_pred, average='micro')
# print('auc score: {}'.format(score))