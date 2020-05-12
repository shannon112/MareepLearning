# TODO: create shell script
# bash train_baseline.sh <trainX_npy> <checkpoint>
echo $1 #: <trainX_npy>
echo $2 #: <checkpoint>
python3 training_strong.py $1 $2 