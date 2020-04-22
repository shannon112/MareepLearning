"""
Created on Mon Nov 21 21:57:29 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
from PIL import Image

import torch
from torch.optim import SGD
from torchvision import models
from torch.autograd import Variable

from utils import readfile
from utils import local_normalize
from model_vgg16_lite import Classifier
from dataset import ImgDataset
from dataset import test_transform

import copy
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, im_path):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = Image.open(im_path).convert('RGB')
        # Hook the layers to get result of the convolution
        self.hook_layer()

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
        for i in range(1, 2001):
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
            #print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image every 20 iteration
            if i % 10 == 0:
                print(self.created_image.shape)
                im_path = './img/ddream_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
        save_image(self.created_image, im_path)


if __name__ == '__main__':
    # THIS OPERATION IS MEMORY HUNGRY! #
    # Because of the selected image is very large
    # If it gives out of memory error or locks the computer
    # Try it with a smaller image
    cnn_layer = 33
    filter_pos = 0

    workspace_dir = sys.argv[1] #'/home/shannon/Downloads/food-11'
    model_filename = sys.argv[2]
    output_dir = sys.argv[3]
    cnnids = [7,14,14,14,24]
    filterids = [0,0,1,2,0]

    #print("Reading data")
    #train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    #print("Size of training data = {}".format(len(train_x)))
    #train_set = ImgDataset(train_x, train_y, test_transform)

    print("Loading model")
    model = Classifier().cuda()
    model.load_state_dict(torch.load(model_filename))

    # showing filters from assigned indices image 
    #img_indices = [800,1602,2001,3201,4001,4800,5600,7000,7400,8003,8801]
    #images, labels = train_set.getbatch(img_indices)


    im_path = '/home/shannon/Downloads/food-11/training/0_1.jpg'
    # Fully connected layer is not needed
    pretrained_model = model.cnn
    dd = DeepDream(pretrained_model, cnn_layer, filter_pos, im_path)
    # This operation can also be done without Pytorch hooks
    # See layer visualisation for the implementation without hooks
    dd.dream()