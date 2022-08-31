import torch

def choose_torch_device() -> str:
    '''Convenience routine for guessing which GPU device to run model on'''
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

    
