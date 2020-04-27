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
    ref_dataset = Adverdataset(images_dirname, label_ids, transform)        
    ref_loader = torch.utils.data.DataLoader(ref_dataset,batch_size = 1,shuffle = False)
    good = 0
    distribution_examples_atk = []
    distribution_examples_ref = []
    ref_labels = []
    pred_labels = []
    atk_labels = []
    for idx,(data, target) in enumerate(atk_loader):
        atk_data, target = data.to(device), target.to(device)
        atk_output = model(atk_data)
        atk_pred = atk_output.max(1, keepdim=True)[1]
        if idx==1 or idx==52 or idx==69 or idx==29:
            softmax = torch.nn.functional.softmax(atk_output, dim = 1)
            distribution_examples_atk.append(softmax.detach().squeeze().cpu().numpy())
            atk_labels.append(atk_pred.item())
        if atk_pred.item() == target.item(): 
            good += 1
            print("good","{:03d}".format(idx),"orig/atk",label_names[target.item()],target.item(),"/",label_names[atk_pred.item()],atk_pred.item())
        else:
            print("fail","{:03d}".format(idx),"orig/atk",label_names[target.item()],target.item(),"/",label_names[atk_pred.item()],atk_pred.item())
    print("atk classification acc: ",good/200,good,"200")
    good = 0
    for idx,(data, target) in enumerate(ref_loader):
        ref_data, target = data.to(device), target.to(device)
        ref_output = model(ref_data)
        ref_pred = ref_output.max(1, keepdim=True)[1]
        if ref_pred.item() == target.item(): 
            good += 1
        if idx==1 or idx==52 or idx==69 or idx==29:
            softmax = torch.nn.functional.softmax(ref_output, dim = 1)
            distribution_examples_ref.append(softmax.detach().squeeze().cpu().numpy())
            pred_labels.append(ref_pred.item())
            ref_labels.append(target.item())
    print("pred classification acc: ",good/200,good,"200")

    # about difference
    img_examples = []
    transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3)])
    ref_dataset = Adverdataset(images_dirname, label_ids, transform)        
    atk_dataset = Adverdataset(output_dirname, label_ids, transform)   
    diff_sum = 0
    for idx,(data, target) in enumerate(atk_dataset):
        if idx==1 or idx==52 or idx==69 or idx==29:
            img_examples.append((data,ref_dataset[idx][0]))
        atk_data = cv2.cvtColor(np.asarray(data),cv2.COLOR_RGB2BGR)  
        ref_data = cv2.cvtColor(np.asarray(ref_dataset[idx][0]),cv2.COLOR_RGB2BGR)
        diff = cv2.norm(ref_data,atk_data, 1)
        diff_sum+=diff
    print("L-inf",diff_sum/200)

    # plot comparison img
    fig, axs = plt.subplots(2, 4, figsize=(40, 6))
    for i,img_set in enumerate(img_examples):
        axs[0][i].imshow(img_set[0])
        axs[0][i].title.set_text("ref:{} {},\npred:{} {},\natk:{} {}\n".format(ref_labels[i],label_names[ref_labels[i]],pred_labels[i],label_names[pred_labels[i]],atk_labels[i],label_names[atk_labels[i]]))
        axs[1][i].imshow(img_set[1])
    plt.setp(axs[0,:], ylabel='attacked')
    plt.setp(axs[1,:], ylabel='reference')
    plt.show()

    # plot comparison distribution
    fig, axs = plt.subplots(2, 4, figsize=(80, 8))
    for i,_ in enumerate(distribution_examples_atk):
        axs[0][i].plot(range(1000),distribution_examples_atk[i])
        sorted_idx = np.argsort(distribution_examples_atk[i])[-3:][::-1]
        axs[0][i].set_title("id/score/name\n{} {:.2f} {},\n{} {:.2f} {},\n{} {:.2f} {},\n".format(
            sorted_idx[0],distribution_examples_atk[i][sorted_idx[0]],label_names[sorted_idx[0]],
            sorted_idx[1],distribution_examples_atk[i][sorted_idx[1]],label_names[sorted_idx[1]],
            sorted_idx[2],distribution_examples_atk[i][sorted_idx[2]],label_names[sorted_idx[2]]
        ),loc='left')
        axs[1][i].plot(range(1000),distribution_examples_ref[i])
        sorted_idx = np.argsort(distribution_examples_ref[i])[-3:][::-1]
        axs[1][i].set_title("id/score/name\n{} {:.2f} {},\n{} {:.2f} {},\n{} {:.2f} {},\n".format(
            sorted_idx[0],distribution_examples_ref[i][sorted_idx[0]],label_names[sorted_idx[0]],
            sorted_idx[1],distribution_examples_ref[i][sorted_idx[1]],label_names[sorted_idx[1]],
            sorted_idx[2],distribution_examples_ref[i][sorted_idx[2]],label_names[sorted_idx[2]]
        ),loc='left')
    plt.setp(axs[0,:], ylabel='attacked',fontsize=20)
    plt.setp(axs[1,:], ylabel='reference',fontsize=20)
    plt.show()

    """
    1 pred fail
    7 pred fail
    8 pred fail
    33 pred fail
    38 pred fail
    46 pred fail
    49 pred fail
    53 pred fail
    83 pred fail
    93 pred fail
    101 pred fail
    106 pred fail
    122 pred fail
    126 pred fail
    137 pred fail
    """