# TODO: create shell script
echo $1 #: testing data
echo $2 #: prediction file
wget -O model/w2v_labeled.model https://www.dropbox.com/s/othu4mqfjq65qb0/w2v_labeled.model?raw=1
wget -O model/last_semi_82.38.model https://www.dropbox.com/s/jl43lpa7tvy5vj8/last_semi_82.38.model?raw=1
python3 predict.py $1 $2