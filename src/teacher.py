import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

@torch.no_grad()
def teacher_normalization(teacher: nn.Sequential, train_loader: DataLoader):
    """
    Args:
        teacher (nn.Sequential): teacher model
        train_loader (DataLoader): training data loader
    
    Returns:
        channel_mean (torch.Tensor): mean of channel
        channel_std (torch.Tensor): standard deviation of channel
    """
    on_gpu = torch.cuda.is_available()
    
    # compute mean
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        train_image = train_image.cuda() if on_gpu else train_image
            
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
        
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    # compute standard deviation
    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        train_image = train_image.cuda() if on_gpu else train_image
            
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std