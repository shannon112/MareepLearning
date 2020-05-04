"""
modified from @author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
from PIL import Image
import copy
import numpy as np
import sys
import matplotlib.pyplot as plt

import torch
from torch.optim import SGD
from torchvision import models
from torch.autograd import Variable

from utils import dd_readfile
from utils import local_normalize
from model_vgg16_lite import Classifier
from dataset import ImgDataset
from dataset import dream_transform

torch.manual_seed(0)

def preprocess_image(pil_im, resize_im=False):
    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")
    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, image,idx,plotid,iterid):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = image
        # Hook the layers to get result of the convolution
        self.hook_layer()
        self.idx = idx
        self.plotid = plotid
        self.iterid = iterid

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, True)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas layer layers need less
        optimizer = SGD([self.processed_image], lr=12,  weight_decay=1e-4)
        im_path = ""

        for i in range(1, self.iterid):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward
                x = layer(x.cuda())
                # Only need to forward until we the selected layer is reached
                if index == self.selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image every 20 iteration
            if i % 10 == 0:
                print(self.created_image.shape)
                im_path = './img/ddream_'+str(self.idx)+'_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
        return self.created_image, im_path

if __name__ == '__main__':
    workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'
    model_filename = sys.argv[2]
    output_dir = sys.argv[3]

    print("Reading data")
    img_indices = [800,1602,2001,3201,4001,4800,5600,7000,7400,8003,8801]
    images = dd_readfile(os.path.join(workspace_dir, "training"), img_indices)

    print("Loading model")
    model = Classifier().cuda()
    model.load_state_dict(torch.load(model_filename))
    pretrained_model = model.cnn    # Fully connected layer is not needed
    print(pretrained_model)

    # different class
    layerid = 30
    iterid = 201
    filterid = 3
    img_indices = [0,8,9]
    fig, axs = plt.subplots(2, len(img_indices), figsize=(30, 20))
    for i, idx in enumerate(img_indices):
        axs[0][i].imshow(images[idx])
    for i, idx in enumerate(img_indices):
        dd = DeepDream(pretrained_model, layerid, filterid, images[idx],0,0,iterid)
        dd_img, dd_path = dd.dream()
        axs[1][i].imshow(dd_img)
    fig.suptitle("deep dream in different class, filter{}, iter{}, layer{}".format(filterid,iterid,layerid),fontsize=36)
    plt.savefig("img/deep_dream_different_class")

    # different filter
    layerid = 30
    iterid = 201
    filterids = [3,12,15]
    fig, axs = plt.subplots(1, len(filterids)+1, figsize=(20, 6))
    axs[0].imshow(images[0])
    axs[0].set_title("original img")
    for i,filterid in enumerate(filterids):
        dd = DeepDream(pretrained_model, layerid, filterid, images[0],0,0,iterid)
        dd_img, dd_path = dd.dream()
        axs[i+1].imshow(dd_img)
        axs[i+1].set_title("filter "+str(filterid))
    fig.suptitle("deep dream in different filter, iter{}, layer{}".format(iterid,layerid),fontsize=24)
    plt.savefig("img/deep_dream_different_filter")

    # different iter
    layerid = 30
    filterid = 3
    iterids = [1,201,401]
    fig, axs = plt.subplots(1, len(iterids), figsize=(20, 6))
    for i,iterid in enumerate(iterids):
        dd = DeepDream(pretrained_model, layerid, filterid, images[0],0,0,iterid)
        dd_img, dd_path = dd.dream()
        axs[i].imshow(dd_img)
        axs[i].set_title("iter "+str(iterid))
    fig.suptitle("deep dream in different iteration, layer{}, filter{}".format(layerid,filterid),fontsize=24)
    plt.savefig("img/deep_dream_different_iter")

    # different layer
    layerids = [3,10,20,30]
    filterid = 3
    iterid = 151
    fig, axs = plt.subplots(1, len(layerids)+1, figsize=(35, 6))
    axs[0].imshow(images[0])
    axs[0].set_title("original img")
    for i,layerid in enumerate(layerids):
        dd = DeepDream(pretrained_model, layerid, filterid, images[0],0,0,iterid)
        dd_img, dd_path = dd.dream()
        axs[i+1].imshow(dd_img)
        axs[i+1].set_title("layer "+str(layerid))
    fig.suptitle("deep dream in different layer, iter{}, filter{}".format(iterid,filterid),fontsize=24)
    plt.savefig("img/deep_dream_different_layer")








