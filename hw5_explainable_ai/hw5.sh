# TODO: create shell script
echo $1 #: training label data
echo $2 #: training unlabel data
wget -O model/w2v_labeled.model https://www.dropbox.com/s/othu4mqfjq65qb0/w2v_labeled.model?raw=1
python3 main.py $1 $2