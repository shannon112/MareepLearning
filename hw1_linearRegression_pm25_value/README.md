# hw1_regression: PM2.5 prediction 
<img src="https://i.imgur.com/d7mPk1R.png" width=300>  
本次作業的資料是從中央氣象局網站下載的真實觀測資料，大家必須利用 linear regression 或其他方法預測 PM2.5 的數值。觀測記錄被分成 train set 跟 test set，前者是每個月的前 20 天所有資料；後者則是從剩下的資料中隨機取樣出來的。Data 含有 18 項觀測數據 AMB_TEMP, CH4, CO, NHMC, NO, NO2, NOx, O3, PM10, PM2.5, RAINFALL, RH, SO2, THC, WD_HR, WIND_DIREC, WIND_SPEED, WS_HR。  

train.csv: 每個月前 20 天的完整資料。  
<img src="https://i.imgur.com/07AYLsj.png" width=600>  
test.csv: 從剩下的 10 天資料中取出240筆資料。每一筆資料都有連續9小時的觀測數據，必須以此預測第10小時的PM2.5。  
<img src="https://i.imgur.com/zF1vK4m.jpg" width=600>  

## Resource
homework video: https://www.youtube.com/watch?v=QfU-qXINCvs&feature=youtu.be  
kaggle: https://www.kaggle.com/c/ml2020spring-hw1/overview  
library limitation: https://reurl.cc/GVkjWD  

## Usage
```
bash  hw1.sh  <input file>  <output file>
bash  hw1_best.sh  <input file>  <output file>
```
## Homework version
https://github.com/NTU-speech-lab/hw1-shannon112
