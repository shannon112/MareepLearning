import sys
import pandas as pd
import numpy as np
import logging

# parameters
input_filename = sys.argv[1]
output_filename = sys.argv[2]
logLevel = sys.argv[3]
modelname = sys.argv[4]
window_len = 5

logging.basicConfig(level=getattr(logging, logLevel), format='%(message)s')
logging.debug(input_filename+" "+output_filename)
logging.debug("\n=\n")

# read from file, 12months*20days as training data, rest as testing data
# read testing data
testdata = pd.read_csv(input_filename, header = None, encoding = 'big5')
logging.debug(testdata)
logging.debug("\n=\n")

# transfer to numpy array with size (4320, 9) = (240candidate*18features, 9hrs)
testdata = testdata.iloc[:, 2:]
testdata[testdata == 'NR'] = 0
test_data = testdata.to_numpy()
logging.debug(test_data)
logging.debug(test_data.shape)
logging.debug("\n=\n")

# transfer to model input format (240candidate, 18features*9hrs)
test_x = np.empty([240, 18*window_len], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), window_len-1:].reshape(1, -1)

# loading pre-train weight, mean, std
mean_x = np.load('weights/'+modelname+'_mean_x.npy')
std_x = np.load('weights/'+modelname+'_std_x.npy')
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
logging.debug(test_x.shape)
logging.debug("\n=\n")

# predict
w = np.load('weights/'+modelname+'_weight.npy')
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
ans_y = np.dot(test_x, w)
logging.debug(ans_y.shape)
logging.debug("\n=\n")

# save as a submit file
import csv
with open(output_filename, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
logging.debug("saved to "+output_filename)
