import torch
from torch import autocast
from contextlib import contextmanager, nullcontext

def choose_torch_device() -> str:
    '''Convenience routine for guessing which GPU device to run model on'''
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def choose_autocast_device(device):
    '''Returns an autocast compatible device from a torch device'''
    device_type = device.type # this returns 'mps' on M1
    # autocast only for cuda, but GTX 16xx have issues with it
    if device_type == 'cuda':
        device_name = torch.cuda.get_device_name()
        if 'GeForce GTX 1660' in device_name or 'GeForce GTX 1650' in device_name:
            return device_type,nullcontext
        else:
            return device_type,autocast
    else:
        return 'cpu',nullcontext
