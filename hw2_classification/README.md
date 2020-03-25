# hw2_classification: binary income classification 

This homework aims to predict whether the income of an individual exceeds $50,000 or not, given some information of the individual. Census-Income (KDD) Dataset is used, with unnecessary attributes removed and the ratio between positively and negatively labeled data balanced.  
https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD)  

There are totally 54256 instances of training data and 27622 instances of test data (13811 in private testing set and 13811 in public testing set). There are 42 features for each row (including id and income). If using the one hot version, there are 512 features for each row (including id and income).  

## Resource
homework video: https://www.youtube.com/watch?v=0_dbrUYoVow&feature=youtu.be  
kaggle: https://www.kaggle.com/c/ml2020spring-hw2/overview  
library limitation: https://reurl.cc/GVkjWD  

## Usage
$1: raw training data (train.csv)  
$2: raw testing data (test_no_label.csv)  
$3: preprocessed training feature (X_train)   
$4: training label (Y_train)  
$5: preprocessed testing feature (X_test)   
$6: output path (prediction.csv)  
```
bash  hw2_logistic.sh $1 $2 $3 $4 $5 $6
bash  hw2_generative.sh $1 $2 $3 $4 $5 $6
bash  hw2_best.sh $1 $2 $3 $4 $5 $6
```

## Homework version
https://github.com/NTU-speech-lab/hw2-shannon112
