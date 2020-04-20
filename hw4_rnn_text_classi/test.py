import torch

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            print(outputs)
            outputs[outputs>=0.5] = 1 # good
            outputs[outputs<0.5] = 0 # bad
            ret_output += outputs.int().tolist()
    return ret_output

def testing_unlabel(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs<0.1] = 0 # bad
            outputs[outputs>=0.9] = -1 # good
            outputs[outputs>=0.1] = 2 # delete
            outputs[outputs==-1] = 1 # good
            ret_output += outputs.int().tolist()
    return ret_output