# TODO: create shell script
#bash hw11_p1.sh <checkpoint> <out_image>
echo $1 #: <checkpoint> 
echo $2 #: <out_image>
python3 test_one.py $1 $2 4