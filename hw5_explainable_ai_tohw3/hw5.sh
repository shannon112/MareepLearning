# TODO: create shell script
echo $1 #: Food dataset directory
echo $2 #: Output images directory
wget -O model/model_vgg16lite https://www.dropbox.com/s/yy7v913fu6vsbot/model_0.9848396501457726_ep150?raw=1
python3 visualize_saliency_map.py $1 model/model_vgg16lite $2
python3 visualize_filter.py $1 model/model_vgg16lite $2
python3 visualize_lime.py $1 model/model_vgg16lite $2
python3 visualize_deepdream.py $1 model/model_vgg16lite $2