# -*- coding: UTF-8 -*- 

import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

from dataset import MyDataset
from dataset import get_dataloader
from model_StudentNet import StudentNet

workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'
model_filename = sys.argv[2]

def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    
    # selected_idx: choosed neuron index in each layer
    selected_idx = []
    # 我們總共有7層CNN，因此逐一抓取選擇的neuron index們。
    for i in range(9):
        # 根據上表，我們要抓的gamma係數在cnn.{i}.1.weight內。
        importance = params[f'cnn.{i}.1.weight']
        # 抓取總共要篩選幾個neuron。
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        # 以Ranking做Index排序，較大的會在前面(descending=True)。
        ranking = torch.argsort(importance, descending=True)
        # 把篩選結果放入selected_idx中。
        selected_idx.append(ranking[:new_dim])

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # 如果是cnn層，則移植參數。
        # 如果是FC層，或是該參數只有一個數字(例如batchnorm的tracenum等等資訊)，那麼就直接複製。
        if name.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # 當處理到Pointwise的weight時，讓now_processed+1，表示該層的移植已經完成。
            if name.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1

            # 如果是pointwise，weight會被上一層的pruning和下一層的pruning所影響，因此需要特判。
            if name.endswith('3.weight'):
                # 如果是最後一層cnn，則輸出的neuron不需要prune掉。
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:,selected_idx[now_processed-1]]
                # 反之，就依照上層和下層所選擇的index進行移植。
                # 這裡需要注意的是Conv2d(x,y,1)的weight shape是(y,x,1,1)，順序是反的。
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:,selected_idx[now_processed-1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    # 讓新model load進被我們篩選過的parameters，並回傳new_model。        
    new_model.load_state_dict(new_params)
    return new_model

valid_dataloader = get_dataloader(workspace_dir,'validation', batch_size=4)
train_dataloader = get_dataloader(workspace_dir,'training', batch_size=4)

net = StudentNet().cuda()
net.load_state_dict(torch.load(model_filename))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = batch_data
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()
        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num

# main training loop
prunning_times = 5
prunning_percentage = 0.95
now_width_mult = 1
for i in range(prunning_times):
    now_width_mult *= prunning_percentage
    new_net = StudentNet(width_mult=now_width_mult).cuda()
    params = net.state_dict()
    net = network_slimming(net, new_net)
    now_best_acc = 0
    for epoch in range(5): # each prunning train 5 epoch
        net.train()
        train_loss, train_acc = run_epoch(train_dataloader, update=True)
        net.eval()
        valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(net.state_dict(), './model/student_custom_small_pruned.bin')
        print('rate {:6.4f} epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(now_width_mult, 
            epoch, train_loss, train_acc, valid_loss, valid_acc))