import os
import torch

print("original cost: {} bytes.".format(os.stat('student_custom_small.bin').st_size))
params = torch.load('student_custom_small.bin')

import numpy as np
import pickle

def encode16(params, fname):
    '''將params壓縮成16-bit後輸出到fname。

    Args:
      params: model的state_dict。
      fname: 壓縮後輸出的檔名。
    '''

    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # 有些東西不屬於ndarray，只是一個數字，這個時候我們就不用壓縮。
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    '''從fname讀取各個params，將其從16-bit還原回torch.tensor後存進state_dict內。

    Args:
      fname: 壓縮後的檔名。
    '''

    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict


encode16(params, '16_bit_model.pkl')
print("16-bit cost: {} bytes.".format(os.stat('16_bit_model.pkl').st_size))

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

encode8(params, '8_bit_model.pkl')
print("8-bit cost: {} bytes.".format(os.stat('8_bit_model.pkl').st_size))