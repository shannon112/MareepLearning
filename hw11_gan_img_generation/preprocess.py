import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from utils import same_seeds

same_seeds(0)

class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img
    def __len__(self):
        return self.num_samples
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def get_dataset(root):
    fnames = sorted(glob.glob(os.path.join(root, '*.jpg')))
    print("dataset images len", len(fnames))
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(), # linearly map [0, 1]
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3), #linearly map [-1, 1]
        ] )
    dataset = FaceDataset(fnames, transform)
    print("A sample in dataset",  dataset[0].shape)
    print("max and min in a image", max(dataset[0].view(-1)),min(dataset[0].view(-1)))
    return dataset


import os
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable

if __name__ == '__main__':
    workspace_dir = sys.argv[1] #~/Downloads/faces/

    # hyperparameters 
    batch_size = 100

    # dataloader (You might need to edit the dataset path if you use extra dataset.)
    dataset = get_dataset(os.path.join(workspace_dir))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # show images
    for i, data in enumerate(dataloader):
        imgs = data
        print(data.shape)
        filename = os.path.join("./img", 'dataset.jpg')
        grid_img = torchvision.utils.make_grid(imgs, nrow=10)
        plt.figure(figsize=(10,8))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.title("dataset randn sample")
        plt.savefig(filename)
        plt.show()
        break