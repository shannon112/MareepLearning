import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import Adverdataset
from torchvision.utils import save_image

device = torch.device("cuda")

class Attacker:
    def __init__(self, img_dir, label):
        # proxy network
        self.model = models.vgg16(pretrained = True)
        #self.model = models.vgg19(pretrained = True)
        #self.model = models.resnet50(pretrained = True)
        #self.model = models.resnet101(pretrained = True)
        #self.model = models.densenet121(pretrained = True)
        #self.model = models.densenet169(pretrained = True)

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
        self.dataset = Adverdataset(img_dir, label, transform)        
        self.loader = torch.utils.data.DataLoader(self.dataset,batch_size = 1,shuffle = False)

    def fgsm_attack(self, image, epsilon, data_grad):
        # find gradient direction
        sign_data_grad = data_grad.sign()
        # add noise
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
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

            # using pre-train model to predict again
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            adv_ex = adv_ex.squeeze().detach().cpu().numpy() 
            perturbed_datas.append(adv_ex)

            # classification right, attacking is fail
            if final_pred.item() == target.item():
                fail += 1
            # classification fail, attacking works
            else:
                success += 1
                #saving
                if len(adv_examples) < 5:
                    data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), data_raw , adv_ex) )
        final_acc = (fail / (wrong + success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return adv_examples, perturbed_datas, final_acc
