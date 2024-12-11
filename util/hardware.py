import torch
from colors import blue, red, highlight


def gpu_check():
    # if gpu is available, print the following
    if torch.cuda.is_available() == True:
        print(f"CUDA available : {blue(torch.cuda.is_available())}")
        print(f"PyTorch version: {highlight(torch.__version__)}")
        print(f"CUDA device count: {highlight(torch.cuda.device_count())}")
        print(f"CUDA current device index: {highlight(torch.cuda.current_device())}")
        print(f"CUDA device name: {highlight(torch.cuda.get_device_name(0))}")
    # if gpu is NOT available, print the following
    else:
        print(f"CUDA available : {red({torch.cuda.is_available()})}")