import numpy as np
import sys
import os
import cv2
import torch 
import matplotlib.pyplot as plt

from utils import readfile
from model_vgg16_lite import Classifier
from dataset import ImgDataset
from dataset import test_transform

# normalize saliency map
def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

# comput saliency map 
def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # 最關鍵的一行 code
    # 因為我們要計算 loss 對 input image 的微分，原本 input x 只是一個 tensor，預設不需要 gradient
    # 這邊我們明確的告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    # saliencies: (batches, channels, height, weight)
    # 因為接下來我們要對每張圖片畫 saliency map，每張圖片的 gradient scale 很可能有巨大落差
    # 可能第一張圖片的 gradient 在 100 ~ 1000，但第二張圖片的 gradient 在 0.001 ~ 0.0001
    # 如果我們用同樣的色階去畫每一張 saliency 的話，第一張可能就全部都很亮，第二張就全部都很暗，
    # 如此就看不到有意義的結果，我們想看的是「單一張 saliency 內部的大小關係」，
    # 所以這邊我們要對每張 saliency 各自做 normalize。手法有很多種，這邊只採用最簡單的
    saliencies = torch.stack([normalize(item) for item in saliencies])
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
    img_indices = [83, 4218, 4707, 8598]
    images, labels = train_set.getbatch(img_indices)
    saliencies = compute_saliency_maps(images, labels, model)
    print(images.shape)
    print(saliencies.shape)

    # plot result
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            # convert tensor from (channels, height, width) to visualable (height, width, channels)
            img_hwc = img.permute(1, 2, 0).numpy()
            # convert img from (BGR) to (RGB)
            img_rgb = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB)
            axs[row][column].imshow(img_rgb)

    figure_name = os.path.join(output_dir, 'saliency_maps_{}_{}_{}_{}.png'.format(img_indices[0],img_indices[1],img_indices[2],img_indices[3]))
    fig.suptitle(figure_name[:-4],fontsize=24)
    plt.savefig(figure_name)
    plt.show()
    plt.close()
