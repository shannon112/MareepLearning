# hw10_anomaly_detection_img: Anomaly Detection on Image
Find the images that their labeled classes do not exist in training set
train = (10000,, 32, 32, 3)  
test = (40000,, 32, 32, 3)  

# Pipeline
image (32, 32, 3) -> CNN/FCN Auto Encoder ->  
latent vector (1536/3) -> Mini-Batch-KMeans (k=3) ->  
Distance to closest center   
reconstructed image (32, 32, 3) ->  
RMSE reconstruction error  
Score = Fusion (Distance, RMSE)  

# Distribution
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/model.png" width=640/>

# Reconstruction
Clustering result: (baseline/improved)  
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p1_baseline.png" width=320/> <img src="https://github.com/shannon112/MareepLearning/blob/master/hw9_unsupervised_img_classi/img/p1_strong.png" width=320/>    

## Resource
homework video: https://www.youtube.com/watch?v=2g5VmgRBiM0&feature=youtu.be  
kaggle: https://www.kaggle.com/c/ml2020spring-hw9/  
library limitation: https://reurl.cc/GVkjWD   

## Usage
```
bash hw10_test.sh <test.npy> <model> <prediction.csv>
bash hw10_train.sh <train.npy> <model>
```

## Homework version
https://github.com/NTU-speech-lab/hw9-shannon112
