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

testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# read img, resize img and get label from filename
def readfile(path, label):    
    img_size = 128
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), img_size, img_size, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(img_size, img_size))
        if label:
          y[i] = int(file.split("_")[0])
    # if label=true: train&valid
    if label:
      return x, y
    # if label=false: test
    else:
      return x