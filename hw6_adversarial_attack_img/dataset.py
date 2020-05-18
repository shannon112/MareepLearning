import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Adverdataset(Dataset):
    def __init__(self, images_dirname, label_ids, transforms):
        self.images_dirname = images_dirname
        self.label_ids = torch.from_numpy(label_ids).long()
        self.transforms = transforms
        self.image_filenames = sorted(os.listdir(images_dirname))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dirname, self.image_filenames[idx]))
        img = self.transforms(img)
        label_ids = self.label_ids[idx]
        return img, label_ids
    
    def __len__(self):
        return len(self.image_filenames)