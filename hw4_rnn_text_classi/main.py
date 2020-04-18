# main.py
import sys
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_training_data
from preprocess import Preprocess
from dataset import TwitterDataset
from model import LSTM_Net
from train import training

# path and filename
path_prefix = "/home/shannon/Downloads/dataset"
model_dir = "./model"
train_w_filename = os.path.join(path_prefix, 'training_label.txt')
train_wo_filename = os.path.join(path_prefix, 'training_label_fixed1.txt')
if len(sys.argv)>2:
    train_w_filename = sys.argv[1]
    train_wo_filename = sys.argv[2]
w2v_model_filename = os.path.join(model_dir, 'w2v_labeled.model')

# checking device to cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

# loading data
print("loading training data ...")
train_x, y = load_training_data(train_w_filename)
train_x_no_label, y_no_label = load_training_data(train_wo_filename)
#train_x = train_x+train_x_no_label
#y = y+y_no_label

# parameters
sen_len = 32#32
fix_embedding = True # fix embedding during training
batch_size = 16#1024
epoch = 15
lr = 0.00005 #0.0002

# preprocessing data
print("preprocessing training data ...")
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_model_filename)
embedding_matrix = preprocess.make_embedding(load=True)
print("embedding_matrix",embedding_matrix.shape)
train_x = preprocess.sentence_word2idx()
print("x",train_x.shape)
y = preprocess.labels_to_tensor(y)
print("y",y.shape)

# model
model = LSTM_Net(embedding_matrix, embedding_dim=250,
                            hidden_dim=150, 
                            num_layers=2, 
                            dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device) # if device is "cuda"，model will use GPU to train（inputs need to be cuda tensor）
model_filename = "./model/ckpt_82.225.model"
model = torch.load(model_filename)

# devide to train90% and vaild10% on labeled training set
X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]
# devide to train90% and vaild10% on both labeled and predicted labeled training set
#X_train, X_val, y_train, y_val = train_x[20000:], train_x[:20000], y[20000:], y[:20000]
# devide to train90% and vaild10% on both labeled and predicted labeled training set
'''
X_train, y_train = train_x[:180000], y[:180000]
X_val, y_val = train_x[180000:200000], y[180000:200000]
X_train = torch.cat((X_train, train_x[200000:]), 0)
y_train = torch.cat((y_train, y[200000:]), 0)
'''

# to dataset
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# to dataloader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# training
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
