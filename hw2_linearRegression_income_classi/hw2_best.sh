# TODO: create shell script
echo $1 #: raw training data (train.csv)
echo $2 #: raw testing data (test_no_label.csv)
echo $3 #: preprocessed training feature (X_train)
echo $4 #: training label (Y_train)
echo $5 #: preprocessed testing feature (X_test)
echo $6 #: output path (prediction.csv)
python3 predict_best.py $5 $6 strong_1224iter