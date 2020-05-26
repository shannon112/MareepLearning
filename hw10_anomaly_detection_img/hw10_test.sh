# TODO: create shell script
#bash hw10_test.sh <test.npy> <model> <prediction.csv>
echo $1 #: <test.npy> 
echo $2 #: <model>
echo $3 #: <prediction.csv>
TAR="${2##*/}"
echo $TAR

if [ "$TAR" = "best.pth" ]; then
    echo "select best model"
    python3 autoencoder_eval_fcn.py $1 $2 $3
elif [ "$TAR" = "baseline.pth" ]; then
    echo "select baseline model"
    python3 autoencoder_eval_cnn.py $1 $2 $3
fi