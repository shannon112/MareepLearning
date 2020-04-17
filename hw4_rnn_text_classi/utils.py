import torch
import numpy as np

# loading training data return list of sentences, sentence is list of words
def load_training_data(path='training_label.txt'):
    # training_label.txt
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines] # words
        y = [line[0] for line in lines] # label
        print("length = ",len(x)) #200000
        return x, y
    # training_nolabel.txt
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        print("length = ",len(x)) #1178614
        return x

# loading testing data return list of sentences, sentence is list of words
def load_testing_data(path='testing_data'):
    # testing_data.txt
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    print("length = ",len(X)) #200000
    return X

def evaluation(outputs, labels):
    # outputs => probability (float), labels => labels
    outputs[outputs>=0.5] = 1 # label as bad
    outputs[outputs<0.5] = 0 # label as good
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct