import os
import numpy as np
import cv2
from PIL import Image

# read img, resize img and get label from filename
def readfile(path, label):    
    img_size = 128
    image_dir = sorted(os.listdir(path))
    print(image_dir[:20])
    print("data len",len(image_dir))
    x = np.zeros((len(image_dir), img_size, img_size, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file)) # read as BGR
        x[i, :, :] = cv2.resize(img,(img_size, img_size))
        if label:
          y[i] = int(file.split("_")[0])
    # if label=true: train&valid
    if label:
      return x, y
    # if label=false: test
    else:
      return x

def dd_readfile(path, img_indices):    
    image_dir = sorted(os.listdir(path))
    print(image_dir[:20])
    print("data len",len(image_dir))
    x_list = []
    for idx in img_indices:
        print(image_dir[idx])
        x = Image.open(os.path.join(path, image_dir[idx])).convert('RGB')
        x_list.append(x)
    return x_list

def local_normalize(image):
    return (image - image.min()) / (image.max() - image.min())