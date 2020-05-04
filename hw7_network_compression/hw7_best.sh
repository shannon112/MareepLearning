# TODO: create shell script
echo $1 #: input dir
echo $2 #: output img dir
python3 main_simple.py $1 $2 0.05 2.0

#validating ./submission/eps0.05m4
#classification acc:  0.075 15 200
#L-inf 5.55
