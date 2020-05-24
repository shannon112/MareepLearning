import sys
import numpy as np
from utils import same_seeds
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

test_filename = sys.argv[1]
same_seeds(0)

# load data
test = np.load(test_filename, allow_pickle=True)
print("data", test.shape)
# input shape
y = test.reshape(len(test), -1)

# plot clustering result
X_embedded = TSNE(n_components=2).fit_transform(y)
print('TSNE for Second Reduction Shape:', X_embedded.shape)
X = X_embedded[:, 0]
Y = X_embedded[:, 1]
plt.scatter(X, Y, s=1)
plt.title("tSNE: testing data raw distribution")
plt.savefig("img/testing_tsne.png")
#plt.show()

