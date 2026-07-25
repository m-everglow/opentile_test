import torch

def is_ampere():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 8 and torch.cuda.get_device_properties(0).minor == 0

def is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9
