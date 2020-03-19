import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# parameters
input_filename = sys.argv[1]
output_filename = sys.argv[2]
logLevel = sys.argv[3]
modelname = sys.argv[4]

logging.basicConfig(level=getattr(logging, logLevel), format='%(message)s')
logging.debug(input_filename+" "+output_filename)
logging.debug("\n=\n")

# parameters
feature_start_index = 0 #9
feature_target_index = 9 #pm2.5 index
#feature_selects = [2,5,8,9,16]
#feature_selects = [0,2,5,7,8,9,11,12,16]
#feature_selects = [0,2,5,7,8,9,10,11,12,16,18,19] #paper
#feature_selects = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17] #w/o RAINFALL 
feature_selects = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19] #w/o RAINFALL with sincos
feature_amount = len(feature_selects) #1
window_len_total = 10
window_len_used = 9
learning_rate = 200
iter_time = 500000 # 500000
experiment_name = "Lr_"+str(learning_rate)+"_Iter_"+str(iter_time)+"_Hr_"+str(window_len_used)+"_Fe_"+str(feature_start_index)+"-"+str(feature_amount)

# read from file, 12months*20days as training data, rest as testing data
# read training data
testdata = pd.read_csv(input_filename, header = None, encoding = 'big5')
logging.debug(testdata)
logging.debug("\n=\n")

# transfer to numpy array with size (4320, 9) = (240candidate*18features, 9hrs)
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
test_data = testdata.to_numpy()
logging.debug(test_data)
logging.debug(test_data.shape)
candidate_num = int(test_data.shape[0] / 18)
logging.debug("\n=\n")

# transfer to model input format (240candidate, 18features*9hrs)
test_x = np.empty([candidate_num, 18*9], dtype = float)
for i in range(candidate_num):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)

logging.debug(test_x.shape)
logging.debug("\n=\n")

# add features (20, 2160)
sample = np.empty([18, 9*candidate_num])
for candidate in range(candidate_num):
    sample[:, candidate * 9 : (candidate + 1) * 9] = test_data[18 * candidate : 18 * (candidate+1), :]
sample = np.concatenate((sample, np.power(np.sin(np.pi*sample[15:16,:]/180),2)), axis = 0).astype(float)
sample = np.concatenate((sample, np.power(np.cos(np.pi*sample[15:16,:]/180),2)), axis = 0).astype(float)
logging.debug(sample.shape)
logging.debug("\n=\n")

# select features (18, 2160)
months_data = None
for idx,feature in enumerate(feature_selects):
    if (idx==0): 
        months_data = sample[feature:feature+1,:]
        continue
    months_data = np.concatenate((months_data, sample[feature:feature+1,:]), axis = 0).astype(float)
logging.debug(months_data.shape)
logging.debug("\n=\n")

# transfer to model input format (240candidate, 18features*9hrs)
test_x = np.empty([candidate_num, feature_amount*9], dtype = float)
for i in range(candidate_num):
    test_x[i, :] = months_data[:, 9 * i :  9 * (i + 1)].reshape(1, -1)
logging.debug(test_x.shape)
logging.debug("\n=\n")

# normalized (mean=0, std=1)
mean_x = np.load('weights/'+modelname+'mean_x.npy')
std_x = np.load('weights/'+modelname+'std_x.npy')
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
            if (test_x[i][j] - mean_x[j])>0.25: test_x[i][j] = mean_x[j] + 0.25
            elif (test_x[i][j] - mean_x[j])<-70: test_x[i][j] = mean_x[j] - 70
        else: 
            print("warning, normalization is not complete.")
logging.debug(test_x.shape)
logging.debug("\n=\n")

# predict
w = np.load('weights/'+modelname+'weight.npy')
test_x = np.concatenate((np.ones([candidate_num, 1]), test_x), axis = 1).astype(float)
ans_y = np.dot(test_x, w)
logging.debug(ans_y.shape)
logging.debug("\n=\n")

# save as a submit file
import csv
with open(output_filename, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(candidate_num):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
logging.info("saved to "+output_filename)