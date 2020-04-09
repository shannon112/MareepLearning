# TODO: create shell script
echo $1 #: raw data directory
echo $2 #: prediction file
wget -O model/model_vgg16lite https://www.dropbox.com/s/yy7v913fu6vsbot/model_0.9848396501457726_ep150?raw=1
python3 predict.py $1 $2 model/model_vgg16lite