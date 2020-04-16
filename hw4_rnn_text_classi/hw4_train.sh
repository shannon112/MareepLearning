# TODO: create shell script
echo $1 #: training label data
echo $2 #: training unlabel data
wget -O model/w2v_best https://www.dropbox.com/s/yy7v913fu6vsbot/model_0.9848396501457726_ep150?raw=1
python3 main.py $1 $2