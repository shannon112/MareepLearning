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
homework video: https://www.youtube.com/watch?v=2g5VmgRBiM0&feature=youtu.be  
kaggle: https://www.kaggle.com/c/ml2020spring-hw9/  
library limitation: https://reurl.cc/GVkjWD   

## Usage
```
bash hw9_best.sh <trainX_npy> <checkpoint> <prediction_path>
bash train_baseline.sh <trainX_npy> <checkpoint>
bash train_best.sh <trainX_npy> <checkpoint>
bash train_improved.sh <trainX_npy> <checkpoint>
```

## Homework version
https://github.com/NTU-speech-lab/hw9-shannon112
