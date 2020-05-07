# -*- coding: UTF-8 -*- 

import torch.nn as nn
import torch.nn.functional as F
import torch

class TeacherNet(nn.Module):

    def __init__(self):
        super(TeacherNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),   
            #128

            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),    
            #64

            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     
            #32

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     
            #16

            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      
            #8
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),     
            #4
        )
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

if __name__ == "__main__":
    teacher_net = TeacherNet().cuda() 
    print(teacher_net)
