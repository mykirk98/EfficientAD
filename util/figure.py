import matplotlib.pyplot as plt
import os

def loss_figure(array:list, dst_fp:str) -> None:
    """
    Create figure from numpy array
    
    array : loss array
    """
    
    file_name = os.path.join(dst_fp, 'loss.png')
    
    # plt.figure(figsize=(10, 10))
    plt.plot(array)
    plt.title(label='Loss')
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel='Loss')
    plt.grid(visible=None)
    plt.savefig(fname=file_name)