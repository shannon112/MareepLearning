import torch
from torch import nn
import torch.optim as optim

from utils import evaluation

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    # print model status
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    # model params
    criterion = nn.BCELoss() #binary cross entropy loss
    t_batch_num = len(train) 
    v_batch_num = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    model.train()
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device is "cuda" inputs to torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device is "cuda" labels to torch.cuda.FloatTensor
            optimizer.zero_grad() # duo to loss.backward(), gradient will accumulate，so we need to zero it every batch
            outputs = model(inputs)
            outputs = outputs.squeeze() # squeeze (batch_size,1) to (batch_size)
            loss = criterion(outputs, labels) # calculate training loss
            loss.backward() # compute gradient from loss
            optimizer.step() # update model parameters

            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] '.format(epoch+1, i+1, t_batch_num), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch_num, total_acc/t_batch_num*100))

        # validation
        model.eval() # set model to eval mode，fix model parameters
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs) 
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch_num, total_acc/v_batch_num*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch_num*100))
        print('-----------------------------------------------')
        model.train() # set model to train mode，let model parameters updatable