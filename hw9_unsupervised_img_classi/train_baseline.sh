# TODO: create shell script
# bash train_baseline.sh <trainX_npy> <checkpoint>
echo $1 #: <trainX_npy>
echo $2 #: <checkpoint>
python3 training_baseline.py $1 $2

#python clustering_baseline.py ~/Downloads/dataset/valX.npy checkpoints/baseline.pth submission
#python evaluation_GT.py ~/Downloads/dataset/valY.npy ./submission/prediction.csv              
