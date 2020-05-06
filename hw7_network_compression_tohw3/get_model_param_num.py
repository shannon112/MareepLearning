import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sys

# model
from model_StudentNet import StudentNet
from model_StudentNet_default import StudentNet

#filename
#model_filename = sys.argv[1]

# testing configuration
model = StudentNet()
model.load_state_dict(torch.load("./model/student_custom_small.bin"))

global_sum = 0
params = model.parameters()
for param in params:
    print(param.size())
    local_sum = 1
    for dim in param.size():
        local_sum = local_sum*dim
    global_sum = global_sum + local_sum
print("total:", global_sum)