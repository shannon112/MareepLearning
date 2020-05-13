# hw9_unsupervised_img_classi: Unsupervised Image Classification
Classify 8500 unlabeled images (32x32x3), see if it is scenery or not.  
trainX = (8500, 32, 32, 3)  
valX = (500, 32, 32, 3)  
valY = (500, 32, 32, 3)  

# Pipeline
image (32, 32, 3) -> Auto Encoder ->  
latent vector (8192) -> PCA-rbf-kernal ->  
reduced vector (200) -> tSNE ->  
reduecd vector (2) -> Mini-Batch-KMeans ->  
two classes  

# Auto Encoder
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/model.png" width=640/>

# Result
Clustering result: (baseline/improved)  
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p1_baseline.png" width=320/> <img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p1_strong.png" width=320/>  
Decoded images from latent vectors: (baseline/improved)  
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p2_baseline.png" width=640/>
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p2_strong.png" width=640/>  
Training Loss v.s. validation accuracy: (baseline/improved)  
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p3_baseline.png" height=320/> <img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p3_strong.png" height=320/>  

## Resource
homework video: https://www.youtube.com/watch?v=Y-a3CZI-wrM&feature=youtu.be  
kaggle: https://www.kaggle.com/c/ml2020spring-hw7/  
library limitation: https://reurl.cc/GVkjWD   

## Usage
```
bash  hw7_test.sh  <data directory>  <prediction file>
```

## Homework version
https://github.com/NTU-speech-lab/hw7-shannon112
