import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# read from file, 12months*20days as training data, rest as testing data
# read training data
data = pd.read_csv('../train.csv', encoding = 'big5')
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
    months_data[months] = sample
    print(months, sample.shape)
print("\n=\n")

# every 10 hrs as a input (9hrs tain, 1hr ans), think in sliding window
x = np.empty([12 * 471, 18 * 9], dtype = float) # get (12month * 471windows , 18features*9hrs)
y = np.empty([12 * 471, 1], dtype = float) # get (12month * 471windows , 18features*1hr)
for months in range(12):
    for days in range(20):
        for hour in range(24):
            if days == 19 and hour > 14: # aborted
                continue
            x[months * 471 + days * 24 + hour, :] = months_data[months][:,days * 24 + hour : days * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[months * 471 + days * 24 + hour, 0] = months_data[months][9, days * 24 + hour + 9] #value
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
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
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
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float) # insert one col in head as constant term
learning_rate = 100
iter_time = 10000 # 500000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
loss_list = []
for t in range(iter_time):
    loss = np.sum(np.power(np.dot(x, w) - y, 2))
    sqrt_loss = np.sqrt(loss/x.shape[0]) #rmse
    loss_list.append(sqrt_loss)
    if(t%1000==0):
        print("iters: {0:<8},loss: {1:<10}".format(str(t), str(sqrt_loss)))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

# saving weights
experiment_name = "Lr_"+str(learning_rate)+"_Iter_"+str(iter_time)
np.save('weights/'+experiment_name+'_mean_x.npy', mean_x)
np.save('weights/'+experiment_name+'_std_x.npy', std_x)
np.save('weights/'+experiment_name+'_weight.npy', w)

# saving learning curve
plt.plot(range(iter_time), loss_list)
plt.suptitle(experiment_name)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('img/'+experiment_name+'.png')
plt.show()