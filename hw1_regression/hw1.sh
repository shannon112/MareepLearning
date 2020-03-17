# TODO: create shell script for running your improved model
# usage: bash ./hw2_best.sh $1 $2

#python predict.py ../hw2_train_val/val1500/images/ ../hw2_train_val/val1500/labelpre/
wget -O model_improved.pth https://www.dropbox.com/s/2qwxal6ebrrxvef/best_improved.pth?raw=1
python3 predict_improved.py $1 $2 model_improved.pth
