# -*- coding: UTF-8 -*- 
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from model_StudentNet import StudentNet
from dataset import ImgDataset
from dataset import testTransform
from dataset import trainTransform
from dataset import readfile

torch.manual_seed(0)
workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'

print("Reading data")
batch_size = 48
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
train_set = ImgDataset(train_x, train_y, trainTransform)
val_set = ImgDataset(val_x, val_y, testTransform)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
print("Size of training data = {}".format(len(train_x)))
print("Size of training data = {}".format(len(val_x)))

# teacher CNN model with acc=88.4, size=47.2MB
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
teacher_net.load_state_dict(torch.load('./teacher_resnet18.bin'))

# student based on Depthwise & Pointwise Convolution Layer instead of regular CNN
student_net = StudentNet(base=16).cuda() 
optimizer = optim.Adam(student_net.parameters(), lr=1e-3)

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # standard Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # KL Divergence of soft_label/T
    soft_loss = nn.KLDivLoss(reduction='batchmean')
        (F.log_softmax(outputs/T, dim=1),  F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        # inputs and soft/hard label data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        with torch.no_grad(): # teacher is only used to get score, do not need to backprop, saving memory
            soft_labels = teacher_net(inputs)
        # training student net
        if update:
            logits = student_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha) # T=20 is from paper
            loss.backward()
            optimizer.step()    
        # validating student net
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
        # calculating loss and acc
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_loss += loss.item() * len(inputs)
        total_num += len(inputs)
    return total_loss / total_num, total_hit / total_num

# main traning loop
teacher_net.eval() # always
now_best_acc = 0
for epoch in range(200):
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)
    # saving best model
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')
    # print result
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))
