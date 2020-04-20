# hw4_rnn_text_classi: Text Sentiment Classification

Dataset is made from the sentences in Twitter posts, if it is positive labeled as 1, negative labeled as 0

Labeled Training set: 200000  
Unlabeled Training set: 1178614  
Testing set: 200000  

## Resource
homework video: https://www.youtube.com/watch?v=P1Lg5l5IPec
kaggle: https://www.kaggle.com/c/ml2020spring-hw4/overview  
library limitation: https://reurl.cc/GVkjWD  

## Model
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw4_rnn_text_classi/img/structure.png" width=560/>

## Usage
```
bash hw4_train.sh <training label data> <training unlabel data>
bash hw4_test.sh <testing data> <prediction file>
```

## Result
<img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw4_rnn_text_classi/img/last_acc.png" width=420/> <img src="https://raw.githubusercontent.com/shannon112/MareepLearning/master/hw4_rnn_text_classi/img/last_loss.png" width=420/>

## Homework version
https://github.com/NTU-speech-lab/hw4-shannon112
