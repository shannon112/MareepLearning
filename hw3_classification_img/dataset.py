import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)

# pytorch dataset
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

normalize = transforms.Normalize(mean=[0.3339, 0.4526, 0.5676],
                                 std=[0.2298, 0.2322, 0.2206])

# testing without data augmentation
test_transform = transforms.Compose([
            transforms.ToPILImage(),                                    
            transforms.CenterCrop(112),
            transforms.ToTensor(),
#            normalize,    # data normalization
])

# training with data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
#    transforms.CenterCrop(112),
    transforms.RandomResizedCrop(112),  # data augmentation
    transforms.RandomHorizontalFlip(),  # data augmentation
    transforms.RandomRotation(60),     # data augmentation 
    transforms.ToTensor(), # image to Tensor，value normalize to [0,1] (data normalization)
#    normalize,   # data normalization
])
