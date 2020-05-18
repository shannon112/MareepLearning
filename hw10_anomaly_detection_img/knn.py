import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from scipy.cluster.vq import vq, kmeans

train_filename = sys.argv[1]
test_filename = sys.argv[2]

train = np.load(train_filename, allow_pickle=True)
test = np.load(test_filename, allow_pickle=True)

x = train.reshape(len(train), -1)
y = test.reshape(len(test), -1)
scores = list()
for n in range(1, 10):
    kmeans_x = MiniBatchKMeans(n_clusters=n, batch_size=100).fit(x)
    y_cluster = kmeans_x.predict(y)
    y_dist = np.sum(np.square(kmeans_x.cluster_centers_[y_cluster] - y), axis=1)

    y_pred = y_dist
    
    #score = f1_score(y_label, y_pred, average='micro')
    #score = roc_auc_score(y_label, y_pred, average='micro')
    #scores.append(score)

#print(np.max(scores), np.argmax(scores))
#print(scores)
#print('auc score: {}'.format(np.max(scores)))