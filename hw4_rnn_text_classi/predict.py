import os
import sys
import pandas as pd
import torch

from utils import load_testing_data
from preprocess import Preprocess
from dataset import TwitterDataset
from model import LSTM_Net
from test import testing

# path and filename
path_prefix = "/home/shannon/Downloads/dataset"
model_dir = "./model"
testing_filename = os.path.join(path_prefix, 'testing_data.txt')
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
model = torch.load(os.path.join(model_dir, 'base_train_082.model'))
outputs = testing(batch_size, test_loader, model, device)

# save result to csv
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv(output_filename, index=False)
print("Finish Predicting")