# hw4_rnn_text_classi: Text Sentiment Classification
Visuallize the CNN network of [hw3_cnn_img_classi](https://github.com/shannon112/MareepLearning/tree/master/hw3_cnn_img_classi)

## Resource
homework video: https://www.youtube.com/watch?v=HFyPZjB-Ex4&feature=youtu.be  
library limitation: https://reurl.cc/GVkjWD  

## 1.Saliency map
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/saliency_maps_800_1602_2001_3201_4001_4800_5600_5600_7000_7400_8003.png" width=840/>

## 2.Filter visualization/activation (gradient ascent)
1<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_visualization_layer7_filter_0.png" width=160/>2<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_visualization_layer14_filter_0.png" width=160/>3<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_visualization_layer14_filter_1.png" width=160/>4<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_visualization_layer14_filter_2.png" width=160/>5<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_visualization_layer24_filter_0.png" width=160/>  
1 <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_activations_layer7_filter0_800_1602_2001_3201_4001_4800_5600_5600_7000_7400.png" width=840/>  
2 <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_activations_layer14_filter0_800_1602_2001_3201_4001_4800_5600_5600_7000_7400.png" width=840/>  
3 <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_activations_layer14_filter1_800_1602_2001_3201_4001_4800_5600_5600_7000_7400.png" width=840/>  
4 <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_activations_layer14_filter2_800_1602_2001_3201_4001_4800_5600_5600_7000_7400.png" width=840/>  
5 <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/filter_activations_layer24_filter0_800_1602_2001_3201_4001_4800_5600_5600_7000_7400.png" width=840/>  
 
## 3.LIME (Local Interpretable Model-Agnostic Explanations)
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/LIME_800_1602_2001_3201_4001_4800_5600_5600_7000_7400_8003.png" width=840/>

## 4.Google Deep Dream (gradient ascent)
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/deep_dream_different_class.png" width=840/>
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/deep_dream_different_iter.png" width=840/>
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/deep_dream_different_filter.png" width=840/>
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw5_explainable_ai/img/deep_dream_different_layer.png" width=840/>

## Execution
```
bash hw5.sh <Food dataset directory> <Output images directory>
```

## Homework version
https://github.com/NTU-speech-lab/hw5-shannon112
