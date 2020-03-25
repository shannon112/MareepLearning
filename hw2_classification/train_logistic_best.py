import numpy as np

np.random.seed(0)
X_train_fpath = '../data/X_train'
Y_train_fpath = '../data/Y_train'
X_test_fpath = '../data/X_test'
output_fpath = '../data/output_{}.csv'

# raw train totoal is 40 + 1 id + 1 y
# processed xtrain totoal is 510 + 1 id + 1 y
# processed ytrain total is 1 id + 1 y
# Parse csv files to numpy array, exclude the col 0 for id (511->510) on X_train
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
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

    '''
    # low pass filter
    if train:
        X_mean = np.mean(X, axis = 0) 
        X_std = np.std(X, axis = 0)  
        print(X_std.shape,X_mean.shape) #510 510
    print(len(X))
    print(len(X[0])) #510
    for i in range(len(X)): 
        for j in range(len(X[0])):  
            X[i][j] = (X[i][j] - X_mean[j]) / (X_std[j]+ 1e-8)
            # filtered by std
            if (X[i][j]) >= 3: X[i][j] = 3
            elif (X[i][j]) <= -3: X[i][j] = -3
    '''
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# select feature
X_train = np.concatenate( (X_train[:,0:366-2],X_train[:,1+494-2:]), axis=1) #country birth
X_train = np.concatenate( (X_train[:,0:221-2],X_train[:,1+277-2:]), axis=1) #previous residence place
X_train = np.concatenate( (X_train[:,0:198-2],X_train[:,1+203-2:]), axis=1) #reason unemploy
#X_train = np.concatenate( (X_train[:,0:193-2],X_train[:,1+194-2:]), axis=1) #gender
#X_train = np.concatenate( (X_train[:,0:178-2],X_train[:,1+182-2:]), axis=1) #race
#X_train = np.concatenate( (X_train[:,0:12-1],X_train[:,82-1:]), axis=1) #recode
#X_train[:,128-2] = np.ceil(X_train[:,128-2] / 1000.) #wage per hour
#X_train = np.concatenate( (X_train, np.power(X_train,2)), axis=1) #square
#X_train = np.concatenate( (X_train, np.sqrt(X_train)), axis=1) #root
X_train_temp = np.concatenate( (np.sqrt(X_train), np.power(X_train,2)), axis=1) #square
X_train = np.concatenate( (X_train, X_train_temp), axis=1) #root
#X_train = np.concatenate( (X_train, np.power(X_train[:,215:216],2)), axis=1) #square certain term
#X_train = np.concatenate( (X_train, np.power(X_train[:,6:7],2)), axis=1) #
#X_train = np.concatenate( (X_train, np.power(X_train[:,222:223],2)), axis=1) #
#X_train = np.concatenate( (X_train, np.power(X_train[:,204:205],2)), axis=1) #
#X_train = np.concatenate( (X_train, np.power(X_train[:,251:252],2)), axis=1) #
#X_train = np.concatenate( (X_train, np.power(X_train[:,243:244],2)), axis=1) #
#X_train = np.concatenate( (X_train, np.power(X_train[:,315:316],2)), axis=1) #
#X_train = np.concatenate( (X_train, np.sqrt(X_train[:,0:1])), axis=1) #
#X_train = np.concatenate( (X_train, np.power(X_train[:,113:114],2)), axis=1) #

X_test = np.concatenate( (X_test[:,0:366-2],X_test[:,1+494-2:]), axis=1)
X_test = np.concatenate( (X_test[:,0:221-2],X_test[:,1+277-2:]), axis=1)
X_test = np.concatenate( (X_test[:,0:198-2],X_test[:,1+203-2:]), axis=1)
#X_test = np.concatenate( (X_test[:,0:193-2],X_test[:,1+194-2:]), axis=1)
#X_test = np.concatenate( (X_test[:,0:178-2],X_test[:,1+182-2:]), axis=1)
#X_test = np.concatenate( (X_test[:,0:12-1],X_test[:,82-1:]), axis=1)
#X_test[:,128-2] = np.ceil(X_test[:,128-2] / 1000.)
#X_test = np.concatenate( (X_test, np.power(X_test,2)), axis=1)
#X_test = np.concatenate( (X_test, np.sqrt(X_test)), axis=1)
X_test_temp = np.concatenate( (np.sqrt(X_test), np.power(X_test,2)), axis=1) #square
X_test = np.concatenate( (X_test, X_test_temp), axis=1) #root
#X_test = np.concatenate( (X_test, np.power(X_test[:,215:216],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,6:7],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,222:223],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,204:205],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,251:252],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,243:244],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,315:316],2)), axis=1) #
#X_test = np.concatenate( (X_test, np.sqrt(X_test[:,0:1])), axis=1) #
#X_test = np.concatenate( (X_test, np.power(X_test[:,113:114],2)), axis=1) #


# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(X_train.shape))
print('Size of development set: {}'.format(X_dev.shape))
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
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b, lamda):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1) + lamda*w
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

# Zero initialization for weights ans bias
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

# Some parameters for training    
max_iter = 3000
batch_size = 1024
learning_rate = 0.2
lamda = 0

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Calcuate the number of parameter updates
step = 1
eps = 0.0000000001
w_adagrad = np.zeros([data_dim])
b_adagrad = 0

# Iterative training
w_best = None
b_best = None
max_dev_acc = 0
min_dev_loss = 1

for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
        
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b,lamda)
            
        # gradient descent update
        # learning rate decay with time

        #adagrad
        w_adagrad += w_grad ** 2
        b_adagrad += b_grad ** 2
        w = w - learning_rate * w_grad / np.sqrt(w_adagrad + eps)
        b = b - learning_rate * b_grad / np.sqrt(b_adagrad + eps)    

        # vanilla gradient decent
        #w = w - learning_rate/np.sqrt(step) * w_grad
        #b = b - learning_rate/np.sqrt(step) * b_grad
        #step = step + 1
            
    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc_now = _accuracy(Y_dev_pred, Y_dev)
    dev_loss_now = _cross_entropy_loss(y_dev_pred, Y_dev) / dev_size
    print("epoch",epoch,"loss",dev_loss_now,"acc",dev_acc_now)
    dev_acc.append(dev_acc_now)
    dev_loss.append(dev_loss_now)
    if dev_loss_now<min_dev_loss:
        min_dev_loss = dev_loss_now
        w_best = w
        b_best = b
        print("min!")

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

import matplotlib.pyplot as plt

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# Predict testing labels
predictions = _predict(X_test, w_best, b_best)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
'''
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
features = features[1:]
features = np.concatenate( (features[0:366-2],features[1+494-2:]), axis=0)
features = np.concatenate( (features[0:221-2],features[1+277-2:]), axis=0)
features = np.concatenate( (features[0:198-2],features[1+203-2:]), axis=0)
print(features.shape)
for i in ind[0:10]:
    print(i,features[i], w[i])
'''