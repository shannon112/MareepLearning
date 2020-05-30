# TODO: create shell script
#bash train_p2.sh <face_dir> <checkpoint>
echo $1 #: <face_dir> 
echo $2 #: <checkpoint>
python3 train_wgan_gp.py $1 $2