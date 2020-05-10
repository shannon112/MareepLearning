# hw7_network_compression: Network Compression on HW3 CNN Based Image Classification Network

Network Compression on HW3 CNN Based Image Classification Network based on Knowledge Distillation, Network Pruning(not complete), Weight Quantiztion, Design Architecture. Dataset is as same as HW3 (Food-11).  

# Depthwise & Pointwise Convolution Layer (MobileNet)
Design Architecture based on Depthwise & Pointwise Convolution Layer, constructing small model.
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw7_network_compression_tohw3/img/Low_Rank_Approximation_Model1.png" width=640/>
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw7_network_compression_tohw3/img/Low_Rank_Approximation_Model2.png" width=640/>

# Knowledge Distillation
Teacher model is ResNet18 with ImageNet pretrained & fine-tune.  
Student model is D&P model above.  
Loss function is:  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw7_network_compression_tohw3/img/Knowledge_Distillation_Loss.png" width=440/>

# Weight Quantiztion
Compress model from default float32 to unit8.  
The encoder mapping function is:  
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw7_network_compression_tohw3/img/Quantization.png" width=340/>

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
