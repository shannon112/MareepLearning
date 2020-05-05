import os
import sys
import numpy as np

#filename
gt_filename = sys.argv[1]
input_filename = sys.argv[2]

# predict
ref = []
with open(gt_filename, 'r') as f:
    f.readline()
    for i, line in  enumerate(f.readlines()):
        label = line.split(',')[1]
        ref.append(int(label))

prediction = []
with open(input_filename, 'r') as f:
    f.readline()
    for i, line in  enumerate(f.readlines()):
        label = line.split(',')[1]
        prediction.append(int(label))

print(len(prediction),len(ref))
print(np.sum(np.array(prediction)==np.array(ref)))
print(np.sum(np.array(prediction)==np.array(ref))/len(prediction))