#!/usr/bin/python
# -*- coding: utf-8 -*-
from torchvision.datasets import ImageFolder

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: image
        """
        sample, target = super().__getitem__(index=index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
            
        Returns:
            sample: image
            target: target of the image
            path: path of the image
        """
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        
        return sample, target, path

def InfiniteDataloader(loader):
    """
    Args:
        loader: DataLoader object
    """
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
