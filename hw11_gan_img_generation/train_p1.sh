# TODO: create shell script
#bash train_p1.sh <face_dir> <checkpoint>
echo $1 #: <face_dir> 
echo $2 #: <checkpoint>
python3 train_dcgan.py $1 $2