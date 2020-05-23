from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class Image_Dataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        if self.transform is not None:
            images = images.transpose(0,2)
            images = self.transform(images)
            images = images*2 -1
        return images

# training with data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    #transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)),
    #transforms.RandomPerspective(),
    #transforms.RandomAffine(15),
    transforms.ToTensor(),
])

# testing without data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])