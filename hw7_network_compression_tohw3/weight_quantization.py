import os
import torch
import numpy as np
import pickle

def encode16(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # there are something not ndarray，just a digit do not need to compress
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param
    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param
    return custom_dict

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

if __name__=="__main__":
    model_filename = sys.argv[1]
    output_dirname = sys.argv[2]
    print("original cost: {} bytes.".format(os.stat(model_filename).st_size))
    params = torch.load(model_filename)

    model_filename_16 = os.path.join(output_dirname,'16_bit_model.pkl')
    encode16(params, os.path.join(model_filename_16))
    print("16-bit cost: {} bytes.".format(os.stat(model_filename_16).st_size))

    model_filename_8 = os.path.join(output_dirname,'8_bit_model.pkl')
    encode8(params, os.path.join(model_filename_8))
    print("8-bit cost: {} bytes.".format(os.stat(model_filename_8).st_size))
