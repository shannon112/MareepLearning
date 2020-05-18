# TODO: create shell script
echo $1 #: input dir
echo $2 #: output img dir
python3 main_simple.py $1 $2 0.1 1.0

#validating ./submission/eps0.1m4
#classification acc:  0.085 17 200
#L-inf 5.55
