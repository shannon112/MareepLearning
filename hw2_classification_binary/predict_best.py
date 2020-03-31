import numpy as np
import sys
X_test_fpath = sys.argv[1]
output_fpath = sys.argv[2]
modelname = sys.argv[3]

# raw train totoal is 40 + 1 id + 1 y
# processed xtrain totoal is 510 + 1 id + 1 y
# processed ytrain total is 1 id + 1 y
# Parse csv files to numpy array, exclude the col 0 for id (511->510) on X_train
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

# select feature
X_test = np.concatenate( (X_test[:,0:366-2],X_test[:,1+494-2:]), axis=1)
X_test = np.concatenate( (X_test[:,0:221-2],X_test[:,1+277-2:]), axis=1)
X_test = np.concatenate( (X_test[:,0:198-2],X_test[:,1+203-2:]), axis=1)
X_test = np.concatenate( (X_test, np.power(X_test,2)), axis=1)
X_test = np.concatenate( (X_test, np.sqrt(X_test)), axis=1)


# Normalize training and testing data
X_mean = np.load('weights/'+modelname+'_mean_x.npy')
X_std = np.load('weights/'+modelname+'_std_x.npy')
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

test_size = X_test.shape[0]
data_dim = X_test.shape[1]
print('Size of testing set: {}'.format(X_test.shape))
print('Dimension of data: {}'.format(data_dim))

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)    

# Iterative training
w_best = np.load('weights/'+modelname+'_weight_w.npy')
b_best = np.load('weights/'+modelname+'_weight_b.npy')

# Predict testing labels
predictions = _predict(X_test, w_best, b_best)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))