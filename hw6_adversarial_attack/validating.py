from fgsm_attack import Attacker

import sys
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import Adverdataset
from torchvision.utils import save_image
from scipy.linalg import norm

if __name__ == '__main__':
    input_dirname = sys.argv[1] #'./submission/images'
    output_dirname = sys.argv[2] #'~/Download/data'
    labels_filename = os.path.join(input_dirname,"labels.csv") 
    categories_filename = os.path.join(input_dirname,"categories.csv") 
    images_dirname = os.path.join(input_dirname,"images")
    device = torch.device("cuda")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # reading labels
    label_ids = pd.read_csv(labels_filename)
    label_ids = label_ids.loc[:, 'TrueLabel'].to_numpy()
    print(len(label_ids),label_ids[0:5],"...")
    label_names = pd.read_csv(categories_filename)
    label_names = label_names.loc[:, 'CategoryName'].to_numpy()
    print(len(label_names),label_names[0:5],"...")

    # proxy network
    #model = models.vgg16(pretrained = True)
    #model = models.vgg19(pretrained = True)
    #model = models.resnet50(pretrained = True)
    #model = models.resnet101(pretrained = True)
    model = models.densenet121(pretrained = True)
    #model = models.densenet169(pretrained = True)
    model.to(device)
    model.eval()

    # about accuracy
    transform = transforms.Compose([                
                    transforms.Resize((224, 224), interpolation=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std, inplace=False)
                ])
    atk_dataset = Adverdataset(output_dirname, label_ids, transform)        
    atk_loader = torch.utils.data.DataLoader(atk_dataset,batch_size = 1,shuffle = False)
    good = 0
    for idx,(data, target) in enumerate(atk_loader):
        atk_data, target = data.to(device), target.to(device)
        atk_output = model(atk_data)
        atk_pred = atk_output.max(1, keepdim=True)[1]
        if atk_pred.item() == target.item(): good += 1
    print("classification acc: ",good/200,good,"200")

    # about difference
    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3)])
    ref_dataset = Adverdataset(images_dirname, label_ids, transform)        
    atk_dataset = Adverdataset(output_dirname, label_ids, transform)   
    diff_sum = 0
    for idx,(data, target) in enumerate(atk_dataset):
        atk_data = cv2.cvtColor(np.asarray(data),cv2.COLOR_RGB2BGR)  
        ref_data = cv2.cvtColor(np.asarray(ref_dataset[idx][0]),cv2.COLOR_RGB2BGR)  
        diff = cv2.norm(ref_data,atk_data, 1)
        diff_sum+=diff
    print("L-inf",diff_sum/200)