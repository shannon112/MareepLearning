import os
import sys
import pandas as pd
import torch

from utils import load_testing_data
from preprocess import Preprocess
from dataset import TwitterDataset
from model import LSTM_Net
#from model_DNN import LSTM_Net
from test import testing_unlabel

# path and filename
path_prefix = "/home/shannon/Downloads/dataset"
model_dir = "./model"
testing_filename = os.path.join(path_prefix, 'training_nolabel.txt')
w2v_model_filename = os.path.join(model_dir, 'w2v_labeled.model')
output_filename = "predict.csv"
if len(sys.argv)>2:
    testing_filename = sys.argv[1]
    output_filename = sys.argv[2]

# checking device to cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

# loading data
print("loading testing data ...")
test_x = load_testing_data(testing_filename)

# parameters
sen_len = 32
batch_size = 16

# predicting
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_model_filename)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()

# to dataset
test_dataset = TwitterDataset(X=test_x, y=None)

# to dataloader
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# testing
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'last_82.43.model'))
outputs = testing_unlabel(batch_size, test_loader, model, device)

# get testing data content
f = open(testing_filename,"r")
lines = f.readlines()

# save label with content to csv
f = open(output_filename, "w")
for i, output in enumerate(outputs):
    if output == 2: continue
    stream = str(output) + " " + "+++$+++" + " " + lines[i].strip() +"\n"
    f.write(stream)
print("save csv ...")
print("Finish Predicting")