# TODO: create shell script
#bash hw10_train.sh <train.npy> <model>
echo $1 #: <train.npy> 
echo $2 #: <model>
TAR="${2##*/}"
echo $TAR

if [ "$TAR" = "best.pth" ]; then
    echo "select best model"
    python3 autoencoder_train_fcn.py $1 $2
elif [ "$TAR" = "baseline.pth" ]; then
    echo "select baseline model"
    python3 autoencoder_train_cnn.py $1 $2
fi