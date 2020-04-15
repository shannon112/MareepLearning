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
        img_size = 112
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),      # [64, 64, 64]
            #img_size /= 2

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),       # [512, 8, 8]
            #img_size /= 2
        )

        img_size = int(img_size/16) # 2^pooling times

        self.fc = nn.Sequential(
            nn.Linear(512*img_size*img_size, 4096),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)