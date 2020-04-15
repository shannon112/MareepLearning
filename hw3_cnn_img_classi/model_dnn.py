import torch
import torch.nn as nn

# if image size is 112, batch can only be 48
#[002/100] 75.14 sec(s) Train Acc: 0.319481 Loss: 0.061322 | Val Acc: 0.430612 loss: 0.053158
torch.manual_seed(0)

# model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dim [3, 128, 128]
        # [3 * 16x7 * 16x7]
        img_size = 112
        self.dnn = nn.Sequential(
            nn.Linear(3*112*112, 64*7*7),
            nn.BatchNorm1d(64*7*7),
            nn.ReLU(),

            nn.Linear(64*7*7, 64*7*7),
            nn.BatchNorm1d(64*7*7),
            nn.ReLU(),
        )

        img_size = int(img_size/16) 

        self.fc = nn.Sequential(
            nn.Linear(64*img_size*img_size, 4096),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Linear(4096, 1024),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 11)
        )

    def forward(self, x):
        out = x.view(x.size()[0], -1)
        out = self.dnn(out)
        return self.fc(out)