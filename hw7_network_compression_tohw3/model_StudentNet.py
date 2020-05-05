# -*- coding: UTF-8 -*- 

import torch.nn as nn
import torch.nn.functional as F
import torch

class StudentNet(nn.Module):
    '''
    Using Depthwise & Pointwise Convolution Layer to build model
    Compare to original Convolution Layer,  Dw&Pw Convolution Layer Accuracy will not drop too many
    It will be used to Knowledge Distillation as a student model
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: the initail model ch (after 3)，and then every layer will base*2，until base*16
            width_mult: for Network Pruning, on base*8 chs Layer, it will * width_mult = pruned ch number
        '''
        super(StudentNet, self).__init__()

        # bandwidth: each Layer's ch
        multiplier = [1, 2, 4, 8, 8, 8, 16, 16, 16]
        bandwidth = [ base * m for m in multiplier]

        # Only pruning the Layer after 3
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # usually we will not grouping in first layer
            # 256
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 128
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 restrict Neuron with min=0 to max=6。 MobileNet series also use ReLU6。
                # it is for the future quantization to float16 / or further qunatization
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # Usually Pointwise Convolution do not need ReLU (or it will be worse)
                nn.MaxPool2d(2, 2, 0),
                # Down Sampling for each block
            ),
            #  64
            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 32
            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 16
            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[7], bandwidth[7], 3, 1, 1, groups=bandwidth[7]),
                nn.BatchNorm2d(bandwidth[7]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[7], bandwidth[8], 1),
            ),

            # Global Average Pooling
            # if images size are inconsistent, it can be uniform to the same, so that it can be used to FC
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(bandwidth[8], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

if __name__ == "__main__":
    student_net = StudentNet(base=16).cuda() 
    print(student_net)
