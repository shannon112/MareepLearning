import numpy as np
import sys
import os
import cv2
import torch 
import matplotlib.pyplot as plt

from utils import readfile
from utils import local_normalize
from model_vgg16_lite import Classifier
from dataset import ImgDataset
from dataset import test_transform

torch.manual_seed(0)

# comput saliency map 
def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()
    # x is default not gradient able, we make x gradient able, so that backward should work, then we can compute x.grad
    x.requires_grad_()
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()
    print("labels",labels)
    print("predictions",y_pred.shape)    
    # saliencies: (batches, channels, height, weight), means that how a small change in x will affect y
    saliencies = x.grad.abs().detach().cpu() # calculate dy/dx
    # each gradient in image is at different scale, self-normalize to make the coloring task easy
    saliencies = torch.stack([local_normalize(item) for item in saliencies])
    return saliencies

if __name__ == "__main__":
    workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'
    model_filename = sys.argv[2]
    output_dir = sys.argv[3]

    print("Reading data")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    train_set = ImgDataset(train_x, train_y, test_transform)

    print("Loading model")
    model = Classifier().cuda()
    model.load_state_dict(torch.load(model_filename))

    # making saliency map from assigned indices image 
    img_indices = [800,1602,2001,3201,4001,4800,5600,7000,7400,8003,8801]
    images, labels = train_set.getbatch(img_indices)
    saliencies = compute_saliency_maps(images, labels, model)
    print(images.shape)
    print(saliencies.shape)

    # plot result
    fig, axs = plt.subplots(2, len(img_indices), figsize=(40, 6))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            # convert tensor from (channels, height, width) to visualable (height, width, channels)
            img_hwc = img.permute(1, 2, 0).numpy()
            # convert img from (BGR) to (RGB)
            img_rgb = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB)
            axs[row][column].imshow(img_rgb)
    figure_name = os.path.join(output_dir, 'saliency_maps_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(
        img_indices[0],img_indices[1],img_indices[2],img_indices[3],img_indices[4],img_indices[5],img_indices[6],
        img_indices[6],img_indices[7],img_indices[8],img_indices[9],img_indices[10]))
    fig.suptitle(figure_name[:-4],fontsize=24)
    plt.savefig(figure_name)
    #plt.show()
    #plt.close()