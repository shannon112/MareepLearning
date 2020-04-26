from fgsm_attack import Attacker

import sys
import os
import pandas as pd
import numpy as np
from PIL import Image

if __name__ == '__main__':
    input_dirname = sys.argv[1] #'./submission/images'
    output_dirname = sys.argv[2] #'~/Download/data'
    labels_filename = os.path.join(input_dirname,"labels.csv") 
    categories_filename = os.path.join(input_dirname,"categories.csv") 
    images_dirname = os.path.join(input_dirname,"images")

    # reading labels
    label_ids = pd.read_csv(labels_filename)
    label_ids = label_ids.loc[:, 'TrueLabel'].to_numpy()
    print(len(label_ids),label_ids[0:5],"...")
    label_names = pd.read_csv(categories_filename)
    label_names = label_names.loc[:, 'CategoryName'].to_numpy()
    print(len(label_names),label_names[0:5],"...")
    image_filenames = sorted(os.listdir(images_dirname))
    print(len(image_filenames),image_filenames[0:5],"...")

    # for hw6 fgsm model
    epsilons = [0.3624] #  with L-inf=19.425
    mIds = [4] # densenet121

    # attacking with diff epsilons
    for eps in epsilons:
        for mId in mIds:
            attacker = Attacker(images_dirname, label_ids,mId)
            imgs = attacker.attack(eps)

            # saving all attacked images
            for img,fn in zip(imgs,image_filenames):
                img[img > 1] = 1
                img[img < 0] = 0
                img = np.round(img * 255)
                img = np.uint8(img).transpose(1, 2, 0)
                im = Image.fromarray(img)
                if not os.path.exists(output_dirname): os.makedirs(output_dirname)
                im.save(os.path.join(output_dirname,str(fn)))