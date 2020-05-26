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
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw10_anomaly_detection_img/img/distribution.png" width=640/>

# Reconstruction
Clustering result: (baseline/improved)  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw10_anomaly_detection_img/img/top2_cnn_reconstruction.png" width=540/> <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw10_anomaly_detection_img/img/top2_fcn_reconstruction.png" width=540/>

## Resource
homework video: https://www.youtube.com/watch?v=gJSwigETXDs&feature=youtu.be  
kaggle: https://www.kaggle.com/c/ml2020spring-hw10/  
library limitation: https://reurl.cc/GVkjWD   

## Usage
```
bash hw10_test.sh <test.npy> <model> <prediction.csv>
bash hw10_train.sh <train.npy> <model>
```

## Homework version
https://github.com/NTU-speech-lab/hw10-shannon112
