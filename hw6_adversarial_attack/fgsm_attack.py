import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import Adverdataset
from torchvision.utils import save_image

device = torch.device("cuda")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

class Attacker:
    def __init__(self, img_dir, label,mId):
        # proxy network
        if mId==0: self.model = models.vgg16(pretrained = True)
        elif mId==1: self.model = models.vgg19(pretrained = True)
        elif mId==2: self.model = models.resnet50(pretrained = True)
        elif mId==3: self.model = models.resnet101(pretrained = True)
        elif mId==4: self.model = models.densenet121(pretrained = True)
        elif mId==5: self.model = models.densenet169(pretrained = True)

        # dataset and dataloader
        self.model.to(device)
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std, inplace=False)
                    ])
        inverseNor = transforms.Compose([ 
                        transforms.Normalize([0,0,0], [1/0.229, 1/0.224, 1/0.225], inplace=False),
                        transforms.Normalize([-0.485, -0.456, -0.406], [1,1,1], inplace=False)
                    ])
        self.dataset = Adverdataset(img_dir, label, transform)        
        self.loader = torch.utils.data.DataLoader(self.dataset,batch_size = 1,shuffle = False)

    def fgsm_attack(self, image, epsilon, data_grad):
        # find gradient direction
        sign_data_grad = data_grad.sign()
        # add noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        adv_examples = []
        perturbed_datas = []
        wrong, fail, success = 0, 0, 0
        for idx,(data, target) in enumerate(self.loader):
            data, target = data.to(device), target.to(device)
            data_raw = data;
            data.requires_grad = True

            # using pre-train model to predict
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # classification fail, do not attack
            if init_pred.item() != target.item():
                wrong += 1
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                perturbed_datas.append(data_raw)
                continue
            
            # classification right, do the attack
            loss = F.nll_loss(output, target) #negiative log-likelihood, a.k.a Cross-Entropy cost function
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            # export attacked img
            adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
            perturbed_datas.append(adv_ex)

        return perturbed_datas
