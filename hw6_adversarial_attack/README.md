# hw6_adversarial_attack: Adversarial Attack on CNN Based Black Box Image Classification Network

Adversarial Attack on CNN Based Black Box Image Classification Network. First, using Fast Gradient Sign Method (FGSM), choosing any proxy network to attack the black box, implement non-targeted FGSM from scratch, tuning your parameter ε, submiting as hw6_fgsm.sh. Second, using any methods you like to attack the model, implementing any methods you prefer from scratch, beating the best performance in hw6_fgsm.sh, beating your classmates with lower L-inf. Norm and higher success rate, submit as hw6_best.sh.  
Black box possible candidate: VGG-16, VGG-19, ResNet-50, ResNet-101, DenseNet-121, DenseNet-169  

<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw6_adversarial_attack/img/cover.png" width=560/>

image set: 200 * (224*224*3) images, 000.png - 199.png  
categories.csv: 1000 categories (0 - 999)  
labels.csv: info of each image  

## Resource
homework video: https://www.youtube.com/watch?v=etW_kpTYetE&feature=youtu.be  
examination website: https://reurl.cc/exvR0R 
library limitation: https://reurl.cc/GVkjWD  
training data: https://reurl.cc/vD3Yr1

## Fast Gradient Sign Method (FGSM) 

## TBD

## Usage
```
timeout 300 bash hw6_fgsm.sh <input dir> <output img dir>
timeout 300 bash hw6_best.sh <input dir> <output img dir>
tar -zcvf <compressed file> <all images>
```

## Result
Average L-inf. norm between all input images and adversarial images  
Success rate of your attack  

## Homework version
https://github.com/NTU-speech-lab/hw6-shannon112
