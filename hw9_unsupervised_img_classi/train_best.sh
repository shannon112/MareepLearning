# TODO: create shell script
echo $1 #: <data directory> 
echo $2 #: <prediction file>
python3 predict_pkl.py $1 $2 model.bin