import torch
import wandb
import argparse
def get_accuracy(logits, targets):
    "Return accuracy given logits and targets."

    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def get_tensor_list_dot(tensor_list1, tensor_list2):
    return torch.stack([tensor1.mul(tensor2).sum() for (tensor1,tensor2) in zip(tensor_list1,tensor_list2)]).sum()

def get_tensor_list_norm(tensor_list):
    return torch.stack([tensor.pow(2).sum() for tensor in tensor_list]).sum().sqrt()

def print_and_write(msg, logfile=None):
    print(msg)
    if logfile is not None:
        print(msg, file=logfile)
        logfile.flush()

def log(log_dict, prefix="", num_step=None):
    new_dict=dict()
    for k in log_dict:
        new_dict[prefix+k]=log_dict[k]
        print(prefix+k, log_dict[k])
    wandb.log(new_dict, step=num_step)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
