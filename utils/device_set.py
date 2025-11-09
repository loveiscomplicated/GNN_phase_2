import torch

def cuda_device_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def mps_device_set():
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def device_set():
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')

    print(f'Using device: {device}')

    return device