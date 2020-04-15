# hw3_classification: Food Classification

This is a dataset containing 16643 food images grouped in 11 major food categories. The 11 categories are Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. Similar as Food-5K dataset, the whole dataset is divided in three parts: training, validation and evaluation. The naming convention is {ClassID}_{ImageID}.jpg, where ID 0-10 refers to the 11 food categories respectively.   

<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/Description.png" width=560/>

Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit.  

Training set: 9866  
Validation set: 3430  
Testing set: 3347  
https://reurl.cc/3DLavL  

## Resource
homework video: https://www.youtube.com/watch?v=L_ebtE4qk14  
kaggle: https://www.kaggle.com/c/ml2020spring-hw3/overview  
library limitation: https://reurl.cc/GVkjWD  

## Model
modified by vgg16  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16lite.png" width=500/>

## Usage
```
bash  hw3_train.sh <data directory>
bash  hw3_test.sh  <data directory>  <prediction file>
```

## Result
train on train 0~100ep  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112_acc.png" width=420/> <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112_loss.png" width=420/>

train on train+val 0~50ep  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112_all_acc.png" width=420/> <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112_all_loss.png" width=420/>

train on train+val 0~50ep  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112_all_acc2.png" width=420/> <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/vgg16_lite_drop_bth48_lr0.002_ep200_deg60_img168_112_all_loss2.png" width=420/>

<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw3_classification_img/img/confusion_matrix.png" width=560/>

## Homework version
https://github.com/NTU-speech-lab/hw3-shannon112
