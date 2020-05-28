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