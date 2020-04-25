from fgsm_attack import Attacker

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    labels_filename = sys.argv[1] #"./data/labels.csv"
    categories_filename = sys.argv[2] #"./data/categories.csv"
    images_dirname = sys.argv[3] #'./data/images'
    output_dirname = sys.argv[4] #'./data/images'

    # reading labels
    label_ids = pd.read_csv(labels_filename)
    label_ids = label_ids.loc[:, 'TrueLabel'].to_numpy()
    print(len(label_ids),label_ids[0:5],"...")
    label_names = pd.read_csv(categories_filename)
    label_names = label_names.loc[:, 'CategoryName'].to_numpy()
    print(len(label_names),label_names[0:5],"...")

    # create Attacker object
    attacker = Attacker(images_dirname, label_ids)
    #epsilons = [10, 1, 0.1, 0.01] #  noise
    epsilons = [0.01] #  noise

    # attacking
    accuracies, examples = [], []
    for eps in epsilons:
        ex, imgs, acc = attacker.attack(eps)
        accuracies.append(acc)
        examples.append(ex)

    # saving all attacked images
    image_filenames = [] #000~199
    for i in range(200):
        image_filenames.append("{:03d}".format(i))
    for img,fn in zip(imgs,image_filenames):
        img[img > 1] = 1
        img[img < 0] = 0
        img = np.round(img * 255)
        img = np.uint8(img).transpose(1, 2, 0)
        im = Image.fromarray(img)
        im.save(os.path.join(output_dirname,str(fn)+".png"))

    # showing attacking result in different noise tolerance
    cnt = 0
    plt.figure(figsize=(30, 8))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,orig_img, ex = examples[i][j]
            # plt.title("{} -> {}".format(orig, adv))
            plt.title("original: {}".format(label_names[orig].split(',')[0]))
            orig_img = np.transpose(orig_img, (1, 2, 0))
            plt.imshow(orig_img)
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]) * 2,cnt)
            plt.title("adversarial: {}".format(label_names[adv].split(',')[0]))
            ex = np.transpose(ex, (1, 2, 0))
            plt.imshow(ex)
    plt.tight_layout()
    plt.savefig("img/result")
    plt.show()