from torchvision import transforms
from parameters import *

basic_transform = transforms.Compose(transforms=[
                        transforms.Resize(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=channel_mean, std=channel_std)
                    ])

penalty_transform = transforms.Compose(transforms=[
                        transforms.Resize(size=(2*image_size, 2*image_size)),
                        transforms.RandomGrayscale(p=0.3),
                        transforms.CenterCrop(size=(image_size, image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=channel_mean, std=channel_std)
                    ])

autoEncoder_transform = transforms.RandomChoice(transforms=[
                                        transforms.ColorJitter(brightness=0.2),
                                        transforms.ColorJitter(contrast=0.2),
                                        transforms.ColorJitter(saturation=0.2)
])