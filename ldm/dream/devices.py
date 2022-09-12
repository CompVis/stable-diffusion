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
    # autocast only supports cuda or cpu
    if device_type in ('cuda','cpu'):
        return device_type,autocast
    else:
        return 'cpu',nullcontext
