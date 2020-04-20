import os
import pandas as pd
from gensim.models import word2vec

from utils import load_training_data
from utils import load_testing_data

# path and filename
path_prefix = "/home/shannon/Downloads/dataset"
train_w_filename = os.path.join(path_prefix, 'training_label.txt')
train_wo_filename = os.path.join(path_prefix, 'training_nolabel.txt')
testing_filename = os.path.join(path_prefix, 'testing_data.txt')

def train_word2vec(x):
    # training word to vector by word embedding
    model = word2vec.Word2Vec(x, size=200, window=5, min_count=5, workers=12, iter=8, sg=1)
    return model

if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data(train_w_filename)
    train_x_no_label = load_training_data(train_wo_filename)

    print("loading testing data ...")
    test_x = load_testing_data(testing_filename)

    #embedding_x = train_x + train_x_no_label + test_x #1578614
    embedding_x = train_x + test_x #400000
    print("embedding 2d list ... lenght=",len(embedding_x))
    model = train_word2vec(embedding_x)
    
    print("saving model ...")
    model.save(os.path.join('model/w2v_200.model'))