import numpy as np
import sys
import os
import torch 
import matplotlib.pyplot as plt
import cv2

from lime import lime_image
from skimage.segmentation import slic

from utils import readfile
from model_vgg16_lite import Classifier
from dataset import ImgDataset
from dataset import test_transform

def predict(input):
    # input: numpy array, (batches, height, width, channels)
    model.eval()
    # input: tensor, (batches, channels, height, width)
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    output = model(input.cuda())
    return output.detach().cpu().numpy()

# segmentation based on skimage library
def segmentation(input):
    return slic(input, n_segments=100, compactness=1, sigma=1)

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

    # showing results from assigned indices image 
    img_indices = [800,1602,2001,3201,4001,4800,5600,7000,7400,8003,8801]
    images, labels = train_set.getbatch(img_indices)
    print(images.shape)

    fig, axs = plt.subplots(2, len(img_indices), figsize=(40, 6))
    np.random.seed(16)

    # plot original images
    for i, img in enumerate(images):
        img_hwc = img.permute(1, 2, 0).numpy() # convert tensor from (channels, height, width) to visualable (height, width, channels)
        img_rgb = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB) # convert img from (BGR) to (RGB)
        axs[0][i].imshow(img_rgb)

    # computing and plot Local Interpretable Model-Agnostic Explanations  (LIME)
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double) # inputs of lime is numpy
        explainer = lime_image.LimeImageExplainer()
        # explainer must include classifier_fn, segmentation_fn
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        # export explainer to image
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
        lime_img, mask = explaination.get_image_and_mask(
                                    label=label.item(),  
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=11,
                                    min_weight=0.05)
        img_rgb = cv2.cvtColor(lime_img.astype(np.float32), cv2.COLOR_BGR2RGB) # convert img from (BGR) to (RGB)
        axs[1][idx].imshow(img_rgb)

    figure_name = os.path.join(output_dir, 'LIME_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        img_indices[0],img_indices[1],img_indices[2],img_indices[3],img_indices[4],img_indices[5],
        img_indices[6],img_indices[6],img_indices[7],img_indices[8],img_indices[9],img_indices[10]))
    plt.savefig(figure_name)
    plt.close()