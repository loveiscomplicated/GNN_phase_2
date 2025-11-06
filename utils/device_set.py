import torch

def cuda_device_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def mps_device_set():
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

