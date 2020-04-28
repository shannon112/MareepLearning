from fgsm_attack import Attacker

import sys
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from dataset import Adverdataset

if __name__ == '__main__':
    input_dirname = sys.argv[1] #'./submission/images'
    output_dirname = sys.argv[2] #'~/Download/data'

    #read imgs
    img_size = 128
    image_fns = sorted(os.listdir(input_dirname))
    print(image_fns[:20])
    print("data len",len(image_fns))

    window_sizes = [3,5,7]
    sigmas = [11,21,31]
    for idx,window_size in enumerate(window_sizes):
        os.makedirs(os.path.join(output_dirname,"avg"+str(window_size)))
        os.makedirs(os.path.join(output_dirname,"gau"+str(window_size)))
        os.makedirs(os.path.join(output_dirname,"med"+str(window_size)))
        os.makedirs(os.path.join(output_dirname,"bil"+str(window_size)))
        for i, file in enumerate(image_fns):
            img = cv2.imread(os.path.join(input_dirname, file)) # read as BGR
            #avg blur
            img_avg = cv2.blur(img,(window_size,window_size))
            cv2.imwrite(os.path.join(output_dirname,"avg"+str(window_size),file),img_avg)

            img_gau = cv2.GaussianBlur(img,(window_size,window_size),0)
            cv2.imwrite(os.path.join(output_dirname,"gau"+str(window_size),file),img_gau)

            img_med = cv2.medianBlur(img,window_size)
            cv2.imwrite(os.path.join(output_dirname,"med"+str(window_size),file),img_med)

            img_bil = cv2.bilateralFilter(img,window_size,sigmas[idx],sigmas[idx])
            cv2.imwrite(os.path.join(output_dirname,"bil"+str(window_size),file),img_bil)
