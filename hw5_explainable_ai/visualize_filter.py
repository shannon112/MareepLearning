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

layer_activations = None
torch.manual_seed(0)

def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    # x: our observation of activation
    # cnnid, filterid: the "cnnid"th layer, and the "filterid"th filter in that layer
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    print("hook-able layer num:",len(model.cnn),"select:",cnnid)
    # tell pytorch to extract certain layer activation map during forward cnn
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)

    # feed x into model and forward it
    model(x.cuda())
    print("raw extracted layer_activations:",layer_activations.shape,"select:",filterid,"in",layer_activations.shape[1])
    # finished extrat filter_activations map
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
        
    # for filter visualization, we can start from random noise or dataset image, here we use later one
    x = x.cuda()
    x.requires_grad_()
    # parameters in optimizer are x instead of model.parameters
    optimizer = torch.optim.Adam([x], lr=lr)
    # gradient ascent to get the x that maximize filter activation
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
        # minimize negativce layer_activations = maximize layer_activations
        objective = -layer_activations[:, filterid, :, :].sum()
        objective.backward() # derive gradient
        optimizer.step() # update parameters
    print("raw maximized layer_activations imgs:",x.shape,"select:",0,"in",x.shape[0])
    filter_visualization = x.detach().cpu().squeeze()[0]
    hook_handle.remove() # reminber to rm it, or it exsits forever in every time forwarding
    return filter_activations, filter_visualization

if __name__ == "__main__":
    workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'
    model_filename = sys.argv[2]
    output_dir = sys.argv[3]
    cnnids = [7,14,14,14,24]
    filterids = [0,0,1,2,0]

    print("Reading data")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    train_set = ImgDataset(train_x, train_y, test_transform)

    print("Loading model")
    model = Classifier().cuda()
    model.load_state_dict(torch.load(model_filename))

    # showing filters from assigned indices image 
    img_indices = [800,1602,2001,3201,4001,4800,5600,7000,7400,8003,8801]
    images, labels = train_set.getbatch(img_indices)

    for i, (cnnid,filterid) in enumerate(zip(cnnids,filterids)):
        filter_activations, filter_visualization = filter_explaination(images, model, cnnid=cnnid, filterid=filterid, iteration=100, lr=0.1)
        print(images.shape)
        print(filter_activations.shape)
        print(filter_visualization.shape)

        # plot filter visualization: what kind of image will maximally activate the filter
        img_hwc = filter_visualization.permute(1, 2, 0).numpy() # convert tensor from (channels, height, width) to visualable (height, width, channels)
        img_rgb = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB) # convert img from (BGR) to (RGB)
        plt.imshow(local_normalize(img_rgb))
        plt.savefig(os.path.join(output_dir,"filter_visualization_layer"+str(cnnid)+"_filter_"+str(filterid)))
        #plt.show()
        plt.close()

        # plot filter activations: the position in images that activate the filter
        fig, axs = plt.subplots(2, len(img_indices), figsize=(40, 6))
        for i, img in enumerate(images):
            img_hwc = img.permute(1, 2, 0).numpy() # convert tensor from (channels, height, width) to visualable (height, width, channels)
            img_rgb = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB) # convert img from (BGR) to (RGB)
            axs[0][i].imshow(img_rgb)
        for i, img in enumerate(filter_activations):
            axs[1][i].imshow(local_normalize(img))
        figure_name = os.path.join(output_dir, 'filter_activations_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            "layer"+str(cnnid),"filter"+str(filterid),img_indices[0],img_indices[1],img_indices[2],img_indices[3],
            img_indices[4],img_indices[5],img_indices[6],img_indices[6],img_indices[7],img_indices[8],img_indices[9],img_indices[10]))
        fig.suptitle(figure_name[:-4],fontsize=24)
        plt.savefig(figure_name)
        #plt.show()
        plt.close()
