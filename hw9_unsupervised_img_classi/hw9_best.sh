# TODO: create shell script
# bash hw9_best.sh <trainX_npy> <checkpoint> <prediction_path>
echo $1 #: <trainX_npy> 
echo $2 #: <checkpoint>
echo $3 #: <prediction_path>
python3 clustering_strong.py $1 $2 $3