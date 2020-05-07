# -*- coding: UTF-8 -*- 
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from dataset import MyDataset
from dataset import get_dataloader

from model_TeacherNet_lite import TeacherNet
#from model_StudentNet_default import StudentNet
from model_FullCnnNet import FullCnnNet
from model_StudentNet_group import StudentNet

torch.manual_seed(0)

workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'

print("Reading data")
train_dataloader = get_dataloader(workspace_dir,'training', batch_size=32)
valid_dataloader = get_dataloader(workspace_dir,'validation', batch_size=32)

#student_net = models.vgg11(pretrained=False, num_classes=11).cuda() 
#student_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda() 
#student_net = FullCnnNet().cuda() 
print(student_net)
#student_net.load_state_dict(torch.load('./model/student_model_deeper_205ep.bin'))
#student_net.load_state_dict(torch.load('./model/student_custom_small.bin'))

optimizer = optim.Adam(student_net.parameters(), lr=1*1e-3)

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        # inputs and soft/hard label data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # training student net
        if update:
            logits = student_net(inputs)
            loss = F.cross_entropy(logits, hard_labels)
            loss.backward()
            optimizer.step()    
        # validating student net
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = F.cross_entropy(logits, hard_labels)
        # calculating loss and acc
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_loss += loss.item() * len(inputs)
        total_num += len(inputs)
    return total_loss / total_num, total_hit / total_num

# main traning loop
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
