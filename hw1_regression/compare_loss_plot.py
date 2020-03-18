import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parameters
training_filename = '../train.csv'
feature_start_index = 0 #9
feature_target_index = 9 #pm2.5 index
feature_amount = 18 #1
window_len = 9
learning_rate = 100
iter_time = 50000 # 500000
experiment_name = "(valid loss) Lr_"+str(learning_rate)+"_Iter_"+str(iter_time)+"_Hr_"+str(window_len)+"_Fe_"+str(feature_start_index)+"-"+str(feature_amount)

# read from file, 12months*20days as training data, rest as testing data
# read training data
data = pd.read_csv(training_filename, encoding = 'big5')
print(data)
print("\n=\n")

# transfer to numpy array with size (4320, 24) = (240days*18features, 24hrs)
data = data.iloc[:, 3:] #row, col
data[data == 'NR'] = 0
raw_data = data.to_numpy()
print(raw_data)
print(raw_data.shape)
print("\n=\n")

# create a dataset(set) with size 12 * (18, 480) = 12months * (18features, 480hr)
months_data = {}
for months in range(12):
    sample = np.empty([18, 480])
    for days in range(20):
        sample[:, days * 24 : (days + 1) * 24] = raw_data[18 * (20 * months + days) : 18 * (20 * months + days + 1), :]
    months_data[months] = sample[feature_start_index:feature_start_index+feature_amount,:]
    print(months, months_data[months].shape)
print("\n=\n")

# every 10 hrs as a input (9hrs tain, 1hr ans), think in sliding window
x = np.empty([12 * 471, feature_amount * window_len], dtype = float) # get (12month * 471windows , 18features*9hrs)
y = np.empty([12 * 471, 1], dtype = float) # get (12month * 471windows , 18features*1hr)
for months in range(12):
    for days in range(20):
        for hour in range(24):
            if days == 19 and hour > 14: # aborted
                continue
            x[months * 471 + days * 24 + hour, :] = months_data[months][:,days * 24 + hour : days * 24 + hour + window_len].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[months * 471 + days * 24 + hour, 0] = months_data[months][feature_target_index, days * 24 + hour + window_len] #value
print(x.shape)
print(y.shape)
print("\n=\n")

# normalized (mean=0, std=1)
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
        else: 
            print("warning, normalization is not complete.")
print("\n=\n")

# split training data to train_set and valid_set
import math
percent_of_usage = 1
x_train_set = x[: math.floor(len(x) * percent_of_usage), :]
y_train_set = y[: math.floor(len(y) * percent_of_usage), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set.shape)
print(y_train_set.shape)
print(x_validation.shape)
print(y_validation.shape)
print("\n=\n")

# training
# model: y = wx, (x = b,x0,x1,x2,...,x162)
# loss: sum((y^ - wx)**2)
# learning rate: w = w - (learning_rate / sqrt(sum(grad**2))) * grad
dim = feature_amount * window_len + 1
x_train_set = np.concatenate((np.ones([x_train_set.shape[0], 1]), x_train_set), axis = 1).astype(float) # insert one col in head as constant term
x_validation = np.concatenate((np.ones([x_validation.shape[0], 1]), x_validation), axis = 1).astype(float)
eps = 0.0000000001
print_gap = 100
learning_rates = [0.01,0.1,1,10,100,1000]
#0.01 iters: 49900   , train_loss: 18.729618155494936, valid_loss: 18.043221717255232 min! 
#0.1 iters: 49900   , train_loss: 5.685879811554643, valid_loss: 5.432788535479192 min! 
#1 iters: 34300   , train_loss: 5.679868125269123, valid_loss: 5.4237776520366126 min! 
#10 iters: 37900   , train_loss: 5.679887534278326, valid_loss: 5.423738930812674 min! 
#100 iters: 34000   , train_loss: 5.6800454206538085, valid_loss: 5.42356064369309 min! 
#1000 iters: 34400   , train_loss: 5.6809708945496675, valid_loss: 5.42247826974859 min! 

for idx, learning_rate in enumerate(learning_rates):
    train_loss_list = []
    valid_loss_list = []
    min_loss = 100
    min_iter = 0
    adagrad = np.zeros([dim, 1])
    w = np.zeros([dim, 1])
    for t in range(iter_time):
        loss = np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))
        if(t%print_gap==0):
            # predict
            ans_y = np.dot(x_validation, w)
            valid_loss = np.sqrt(np.sum(np.power(ans_y - y_validation,2))/x_validation.shape[0])
            valid_loss_list.append(valid_loss)
            train_loss = np.sqrt(loss/x_train_set.shape[0]) #rmse
            train_loss_list.append(train_loss)
            if valid_loss<min_loss: 
                select_flag = "min!"
                min_loss = valid_loss
                min_iter = t
            else: select_flag = ""
            print("iters: {0:<8}, train_loss: {1:<10}, valid_loss: {2:<10} {3:<5}".format(str(t), str(train_loss), str(valid_loss),select_flag))
        gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) #dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    #plt.plot(range(0,iter_time,print_gap), train_loss_list,label=str(learning_rate), c=plt.cm.Set2(idx))
    min_valid_loss = "min_valid_loss = {0:},{1:.5f}".format(min_iter, min_loss)
    plt.plot(min_iter, min_loss,'o',label=min_valid_loss, c=plt.cm.Set2(idx))
    plt.plot(range(0,iter_time,print_gap), valid_loss_list,label=str(learning_rate), c=plt.cm.Set2(idx))
 
# saving learning curve
plt.legend()
plt.suptitle(experiment_name)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('img/'+experiment_name+'.png')
plt.show()